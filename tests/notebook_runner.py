"""
Run a LASSI notebook's own code, headless, with a scripted stand-in for the LLM.

Used by the reproducibility test: the original v0 notebook and the current notebook are
run under identical conditions, and every prompt each one sends is recorded so the two
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

# Order of the values returned by experimental_setup() in the notebooks
_SETUP_FIELDS = ("app_folder_name", "source_lang", "target_lang", "model", "num_ctx", "llm_source",
                 "use_context", "context_knowledge_selection", "code_compiler", "code_compiler_kwds",
                 "yes_execute", "exe_input_parameters", "experiment", "display_response",
                 "test_with_error", "generate_test_case",
                 "use_judge", "model_judge", "record_results", "total_trials")

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


def run_notebook(notebook_path, app="layout", app_args=("1",), source_lang="CUDA",
                 target_lang="OMP", overrides=None):
    """
    Execute `notebook_path` end to end in a temporary working copy.

    Returns a dict:
        transcript  list of (system_prompt, content_prompt) in the order they were sent
        files       {relative_path: text} for the generated source files
        completed   True if the pipeline reported success
    "defaults" holds experimental_setup()'s own return values, before the test's fixed config is applied.
    `overrides` maps experimental_setup() return-value names to values (new notebook only).
    """
    _stub_missing_modules()
    nb = json.load(open(notebook_path))
    code = [re.sub(r"^\s*%.*$", "", "".join(c["source"]), flags=re.M)
            for c in nb["cells"] if c["cell_type"] == "code"]
    is_new = "llm_backends" in code[0]

    src_dir = os.path.join(REPO, "translated_code", "input_codes", "HeCBench", app)
    ref_cpp = open(os.path.join(src_dir, f"{app}-{target_lang.lower()}_main.cpp")).read()

    transcript, answers = [], iter([_BAD_COMPILE, _BAD_EXECUTE, "```cpp\n" + ref_cpp + "\n```"])

    def scripted_llm(system_prompt, content_prompt):
        transcript.append((system_prompt, content_prompt))
        if "summary description the following background knowledge" in content_prompt:
            return "BACKGROUND SUMMARY"
        if "summary description of the following code" in content_prompt:
            return "CODE DESCRIPTION"
        return next(answers)

    with tempfile.TemporaryDirectory(prefix="lassi_test_") as work:
        dst = os.path.join(work, "translated_code", "input_codes", "HeCBench", app)
        shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns("*_main"))
        shutil.copy(os.path.join(REPO, "prompt_dictionary.py"), work)
        if is_new:
            shutil.copy(os.path.join(REPO, "llm_backends.py"), work)

        old_cwd, old_path = os.getcwd(), list(sys.path)
        os.chdir(work); sys.path.insert(0, work)
        for mod in ("prompt_dictionary", "llm_backends"):
            sys.modules.pop(mod, None)
        try:
            g = {"__name__": "__main__"}
            with contextlib.redirect_stdout(io.StringIO()):
                for src in code[:-1]:   # everything except the final "run" cell
                    exec(src, g)

                # --- stand-in LLM -------------------------------------------------
                if is_new:
                    import llm_backends
                    llm_backends.call_llm = lambda src, model, ctx, sys_p, user_p, *a, **k: scripted_llm(sys_p, user_p)
                    llm_backends.unload_ollama_model = lambda model: None
                else:
                    g["ollama_llm"] = lambda model, ctx, sys_p, user_p: scripted_llm(sys_p, user_p)
                    g["unload_ollama_model"] = lambda model: None

                # --- fixed experiment configuration (the paper's defaults otherwise) --
                base = (app, source_lang, target_lang, "scripted-model", 4096, "ollama", True,
                        target_lang.lower(), "g++", "-fopenmp -O1", True, list(app_args),
                        "test", False, False, False)
                if is_new:
                    ov = {"use_judge": False, "model_judge": "scripted-model",
                          "record_results": False, "total_trials": 1}
                    ov.update(overrides or {})
                    base += (ov["use_judge"], ov["model_judge"], ov["record_results"], ov["total_trials"])
                # Record the notebook's own defaults (what a user gets without editing anything)
                real_defaults = dict(zip(_SETUP_FIELDS, g["experimental_setup"]()))
                g["experimental_setup"] = lambda: base
                exec(code[-1], g)

            files = {}
            for root, _, names in os.walk(os.path.join(work, "translated_code")):
                for n in names:
                    if n.endswith(".cpp") and "input_codes" not in root:
                        p = os.path.join(root, n)
                        files[os.path.relpath(p, work)] = open(p).read()
            return {"transcript": transcript, "files": files, "completed": g.get("complete", False),
                    "defaults": real_defaults}
        finally:
            os.chdir(old_cwd); sys.path[:] = old_path
