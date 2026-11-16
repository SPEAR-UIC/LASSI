"""
Run LASSI headless with a scripted stand-in for the LLM.

Used by the reproducibility test: the ORIGINAL notebook (tests/legacy) and the CURRENT package
are run under identical conditions, and every prompt each one sends is recorded so the two
transcripts can be compared. Only g++ is needed (no GPU, no model, no network).
"""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tests must never pick up a developer's own config.yaml (it may hold a real API key): an empty
# LASSI_CONFIG means "use no config file". The config-file tests set it themselves.
USER_LASSI_CONFIG = os.environ.get("LASSI_CONFIG")      # what the person running the tests had set, if anything
os.environ["LASSI_CONFIG"] = ""

# Order of the values returned by experimental_setup() in the original notebook
_SETUP_FIELDS = ("app_folder_name", "source_lang", "target_lang", "model", "num_ctx", "llm_source",
                 "use_context", "context_knowledge_selection", "code_compiler", "code_compiler_kwds",
                 "yes_execute", "exe_input_parameters", "experiment", "display_response",
                 "test_with_error", "generate_test_case")

_BAD_COMPILE = "```cpp\nint main() { return undefined_symbol_xyz; }\n```"
_BAD_EXECUTE = '```cpp\n#include <cstdio>\nint main() { fprintf(stderr, "boom\\n"); return 3; }\n```'


def _stub_missing_modules():
    """Provide minimal stand-ins for optional third-party modules that are not installed."""
    def need(name):
        try:
            __import__(name)
            return False
        except ImportError:
            return True

    if need("tiktoken"):
        m = types.ModuleType("tiktoken")

        class _Enc:
            _pat_str, _mergeable_ranks, _special_tokens = "", {}, {}
            def encode(self, t, **k): return list(t)
            def decode(self, t): return "".join(t)
        m.get_encoding = lambda name: _Enc()
        m.Encoding = lambda *a, **k: _Enc()
        sys.modules["tiktoken"] = m
    if need("IPython"):
        ip = types.ModuleType("IPython"); disp = types.ModuleType("IPython.display")
        disp.display = lambda *a, **k: None
        disp.Markdown = str
        disp.HTML = str
        ip.display = disp
        sys.modules["IPython"], sys.modules["IPython.display"] = ip, disp
    if need("openai"):
        m = types.ModuleType("openai"); m.OpenAI = object; sys.modules["openai"] = m
    if need("ollama"):
        m = types.ModuleType("ollama"); m.ResponseError = Exception; sys.modules["ollama"] = m


def _scripted_llm(app, target_lang, scenario="errors"):
    """
    Return (transcript list, function(system_prompt, content_prompt) -> reply).
    scenario "errors": a compile error, then an execution error, then good code (once).
    scenario "clean":  good code every time, and a VALID verdict when asked to judge.
    """
    ext = "cu" if target_lang in ("CUDA", "HIP") else "cpp"
    ref_cpp = open(os.path.join(REPO, "translated_code", "input_codes", "HeCBench", app,
                                f"{app}-{target_lang.lower()}_main.{ext}")).read()
    good = "```cpp\n" + ref_cpp + "\n```"
    transcript = []
    answers = iter([_BAD_COMPILE, _BAD_EXECUTE, good]) if scenario == "errors" else iter(lambda: good, None)

    def llm(system_prompt, content_prompt):
        transcript.append((system_prompt, content_prompt))
        if "summary description the following background knowledge" in content_prompt:
            return "BACKGROUND SUMMARY"
        if "summary description of the following code" in content_prompt:
            return "CODE DESCRIPTION"
        if "conclude your response with VALID" in content_prompt:
            return "The outputs agree and the logic matches, so the translation is valid.\nVALID"
        return next(answers)
    return transcript, llm


def _collect_outputs(work, suffix=".cpp"):
    files = {}
    for root, _, names in os.walk(os.path.join(work, "translated_code")):
        for n in names:
            if n.endswith(suffix) and "input_codes" not in root:
                p = os.path.join(root, n)
                files[os.path.relpath(p, work)] = open(p).read()
    return files


def _collect_all(work):
    """Every text file produced under translated_code/ (not the input sources), by relative path."""
    out = {}
    for root, _, names in os.walk(os.path.join(work, "translated_code")):
        if "input_codes" in root:
            continue
        for n in names:
            p = os.path.join(root, n)
            try:
                out[os.path.relpath(p, work)] = open(p).read()
            except UnicodeDecodeError:      # compiled binary
                pass
    return out


def _workdir(work, app, modules_from=None):
    """A scratch working copy: the app's source folder (no compiled binaries), plus legacy modules."""
    src_dir = os.path.join(REPO, "translated_code", "input_codes", "HeCBench", app)
    dst = os.path.join(work, "translated_code", "input_codes", "HeCBench", app)
    shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns("*_main"))
    if modules_from:
        for f in os.listdir(modules_from):
            if f.endswith(".py"):
                shutil.copy(os.path.join(modules_from, f), work)


class _Sandbox:
    def __init__(self, work):
        self.work = work
    def __enter__(self):
        self.old_cwd, self.old_path = os.getcwd(), list(sys.path)
        os.chdir(self.work)
        return self
    def __exit__(self, *exc):
        os.chdir(self.old_cwd); sys.path[:] = self.old_path


def run_legacy(notebook_path, app="layout", app_args=("1",), source_lang="CUDA", target_lang="OMP"):
    """Execute the original notebook's own code end to end in a temporary working copy."""
    _stub_missing_modules()
    nb = json.load(open(notebook_path))
    code = [re.sub(r"^\s*%.*$", "", "".join(c["source"]), flags=re.M)
            for c in nb["cells"] if c["cell_type"] == "code"]
    transcript, llm = _scripted_llm(app, target_lang)

    with tempfile.TemporaryDirectory(prefix="lassi_legacy_") as work:
        _workdir(work, app, modules_from=os.path.join(REPO, "tests", "legacy"))
        with _Sandbox(work):
            sys.path.insert(0, work)
            sys.modules.pop("prompt_dictionary", None)
            g = {"__name__": "__main__"}
            with contextlib.redirect_stdout(io.StringIO()):
                for src in code[:-1]:           # everything except the final "run" cell
                    exec(src, g)
                g["ollama_llm"] = lambda model, ctx, sys_p, user_p: llm(sys_p, user_p)
                g["unload_ollama_model"] = lambda model: None
                # The notebook's own defaults, before the test pins its fixed configuration
                defaults = dict(zip(_SETUP_FIELDS, g["experimental_setup"]()))
                fixed = (app, source_lang, target_lang, "scripted-model", 4096, "ollama", True,
                         target_lang.lower(), "g++", "-fopenmp -O1", True, list(app_args),
                         "test", False, False, False)
                g["experimental_setup"] = lambda: fixed
                exec(code[-1], g)
            return {"transcript": transcript, "files": _collect_outputs(work),
                    "completed": g.get("complete", False), "defaults": defaults}


def run_current(app="layout", app_args=("1",), package_root=None, scenario="errors", **config_overrides):
    """Run the lassi package (from `package_root`, default: this repo) end to end in a temporary working copy."""
    _stub_missing_modules()
    transcript, llm = _scripted_llm(app, config_overrides.get("target_lang", "OMP"), scenario)

    with tempfile.TemporaryDirectory(prefix="lassi_current_") as work:
        _workdir(work, app)
        with _Sandbox(work):
            sys.path.insert(0, package_root or REPO)
            for m in [m for m in sys.modules if m == "lassi" or m.startswith("lassi.")]:
                del sys.modules[m]
            import lassi
            from lassi import llm_backends
            llm_backends.call_llm = lambda src, model, ctx, sys_p, user_p, *a, **k: llm(sys_p, user_p)
            llm_backends.unload_ollama_model = lambda model: None

            defaults = lassi.Config()          # what a user gets without changing anything
            kwargs = dict(app=app, model="scripted-model", num_ctx=4096, compiler="g++",
                          compiler_flags="-fopenmp -O1", exe_params=list(app_args),
                          experiment="test", display_response=False)
            kwargs.update(config_overrides)
            cfg = lassi.Config(**kwargs)
            with contextlib.redirect_stdout(io.StringIO()):
                outcomes = lassi.run_experiment(cfg)
            return {"transcript": transcript, "files": _collect_outputs(work),
                    "completed": bool(outcomes) and all(outcomes), "defaults": defaults,
                    "csv": _collect_outputs(work, ".csv"), "metadata": _collect_outputs(work, ".txt"),
                    "all_files": _collect_all(work)}


def run_cli(argv, app="layout", scenario="errors"):
    """Run `python run_lassi.py <argv>` in-process in a temporary working copy, with the scripted LLM."""
    _stub_missing_modules()
    transcript, llm = _scripted_llm(app, "OMP", scenario)

    with tempfile.TemporaryDirectory(prefix="lassi_cli_") as work:
        _workdir(work, app)
        with _Sandbox(work):
            sys.path.insert(0, REPO)
            for m in [m for m in sys.modules if m == "lassi" or m.startswith("lassi.")]:
                del sys.modules[m]
            from lassi import llm_backends
            from lassi.cli import main
            llm_backends.call_llm = lambda src, model, ctx, sys_p, user_p, *a, **k: llm(sys_p, user_p)
            llm_backends.unload_ollama_model = lambda model: None
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    code = main(argv)
                except SystemExit as e:        # argparse exits (help, usage errors)
                    code = e.code
            return {"exit_code": code, "output": out.getvalue(), "transcript": transcript,
                    "files": _collect_outputs(work), "csv": _collect_outputs(work, ".csv"),
                    "all_files": _collect_all(work)}
