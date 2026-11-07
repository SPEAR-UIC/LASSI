"""
Reproducibility test: with its default configuration, the current LASSI must behave
exactly like the original notebook that produced the paper's results.

The original notebook and the current package are run on the same small kernel with a scripted LLM that forces one compile
error and one execution error, so initial, compile-correction and execution-correction
prompts are all exercised. Every prompt sent to the LLM must be identical, and so must the
final code that is written. Requires only g++ (no GPU, model or network).

Run:  python tests/test_reproducibility.py      (or: pytest tests)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, _stub_missing_modules, run_cli, run_current, run_legacy

LEGACY = os.path.join(REPO, "tests", "legacy", "LASSI_pipeline_v0_original.ipynb")


def _first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return min(len(a), len(b)), None, None


# Settings that define "paper mode". Compared between the original and the current notebook as
# shipped (machine-specific values such as app, compiler path and model are deliberately excluded).
PAPER_DEFAULTS = ("source_lang", "target_lang", "use_context", "llm_source", "num_ctx",
                  "yes_execute", "display_response", "test_with_error", "generate_test_case")
# Optional features added after the paper: must all be off by default.
OPTIONAL_OFF = {"use_judge": False, "record_results": False, "total_trials": 1}
LEGACY_NAMES = {"source_lang": "source_lang", "target_lang": "target_lang", "use_context": "use_context",
                "llm_source": "llm_source", "num_ctx": "num_ctx", "yes_execute": "yes_execute",
                "display_response": "display_response", "test_with_error": "test_with_error",
                "generate_test_case": "generate_test_case"}


def test_default_config_matches_original():
    assert shutil.which("g++"), "g++ is required for this test"
    legacy = run_legacy(LEGACY)
    current = run_current()

    for name in PAPER_DEFAULTS:
        original, now = legacy["defaults"][LEGACY_NAMES[name]], getattr(current["defaults"], name)
        assert original == now, f"default {name!r} changed: original {original!r}, current {now!r}"
    for name, off in OPTIONAL_OFF.items():
        assert getattr(current["defaults"], name) == off, f"optional feature {name!r} is not off by default"

    assert legacy["completed"] and current["completed"], "pipeline did not complete"
    assert len(legacy["transcript"]) == 5, (   # summary, description, initial, 2 corrections
        f"unexpected scenario: {len(legacy['transcript'])} LLM calls in the original")
    assert len(current["transcript"]) == len(legacy["transcript"]), (
        f"LLM call count differs: original {len(legacy['transcript'])}, current {len(current['transcript'])}")

    i, a, b = _first_difference(legacy["transcript"], current["transcript"])
    assert a is None, f"prompt #{i} differs between original and current:\n--- original\n{a}\n--- current\n{b}"

    assert list(legacy["files"].values()) == list(current["files"].values()), "generated code differs"


def test_prompt_dictionary_matches_original():
    """
    Every system prompt, translation prompt and language-knowledge text, for every direction
    (including OMP->CUDA, which the end-to-end test cannot run without nvcc), is identical.
    """
    import importlib.util
    _stub_missing_modules()
    spec = importlib.util.spec_from_file_location(
        "legacy_prompt_dictionary", os.path.join(REPO, "tests", "legacy", "prompt_dictionary.py"))
    legacy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_mod)
    sys.path.insert(0, REPO)
    from lassi.prompt_dictionary import PromptDictionary

    old, new = legacy_mod.PromptDictionary(), PromptDictionary()
    for attr in sorted(vars(old)):
        assert getattr(old, attr) == getattr(new, attr), f"PromptDictionary.{attr} differs from the original"


def test_optional_features():
    """The opt-in judge, results CSV and repeated trials work, and leave the default path alone."""
    import csv
    import io
    assert shutil.which("g++"), "g++ is required for this test"
    r = run_current(scenario="clean", use_judge=True, model_judge="scripted-judge",
                    record_results=True, total_trials=2)
    assert r["completed"], "pipeline did not complete"

    # one trial = context summary + code description + generation (+ judge)  ->  4 calls per trial
    assert len(r["transcript"]) == 8, f"expected 8 LLM calls for 2 judged trials, got {len(r['transcript'])}"

    names = sorted(os.path.basename(p) for p in r["files"])
    assert any("_Trial1" in n for n in names) and any("_Trial2" in n for n in names), names

    (csv_text,) = r["csv"].values()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [row["trial_number"] for row in rows] == ["1", "2"], rows
    for row in rows:
        assert row["lassi_judge_valid"] == "VALID", row
        assert row["lassi_compiled"] == "Y" and row["lassi_execute"] == "Y", row
        assert float(row["target_orig_runtime_s"]) > 0 and float(row["lassi_runtime_s"]) > 0, row
        assert float(row["lassi_execute_ratio"]) > 0, row
    assert any("LLM-as-a-Judge result: VALID" in t for t in r["metadata"].values())

    # With the features off (the default) none of that appears.
    d = run_current(scenario="clean")
    assert len(d["transcript"]) == 3 and not d["csv"], "optional features leaked into the default run"
    assert not any("Judge" in t for t in d["metadata"].values())


# Machine-specific settings the tests must pin (the paper's compiler needs nvc++ / a GPU)
_CLI_BASE = ["--app", "layout", "--llm-source", "ollama", "--model", "scripted-model", "--num-ctx", "4096",
             "--compiler", "g++", "--compiler-flags", "-fopenmp -O1", "--exe-params", "1",
             "--experiment", "test", "--quiet"]


def test_cli_matches_original():
    """`run_lassi.py` with no optional flags sends exactly the prompts the original notebook did."""
    assert shutil.which("g++"), "g++ is required for this test"
    legacy = run_legacy(LEGACY)
    cli = run_cli(_CLI_BASE)
    assert cli["exit_code"] == 0, cli["output"][-500:]
    assert cli["transcript"] == legacy["transcript"], "CLI prompts differ from the original"
    assert list(cli["files"].values()) == list(legacy["files"].values()), "CLI generated code differs"


def test_cli_options_and_errors():
    import csv
    import io
    assert shutil.which("g++"), "g++ is required for this test"

    r = run_cli(_CLI_BASE + ["--judge", "--judge-model", "scripted-judge", "--record-results", "--trials", "2"],
                scenario="clean")
    assert r["exit_code"] == 0, r["output"][-500:]
    (csv_text,) = r["csv"].values()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 2 and all(row["lassi_judge_valid"] == "VALID" for row in rows), rows

    # usage errors exit non-zero with a message, and never run the pipeline
    bad = run_cli(["--app", "no-such-app", "--compiler", "g++"], scenario="clean")
    assert bad["exit_code"] == 2 and "Unknown app" in bad["output"] and not bad["transcript"], bad["output"]
    assert run_cli(["--source", "CUDA", "--target", "CUDA"], scenario="clean")["exit_code"] == 2
    assert run_cli(["--trials", "0"] + _CLI_BASE, scenario="clean")["exit_code"] == 2
    assert run_cli(["--help"])["exit_code"] == 0

    listing = run_cli(["--list-apps"])
    assert listing["exit_code"] == 0 and "randomAccess" in listing["output"]


def test_notebook_ships_paper_defaults():
    """
    The notebook's config cell, run as shipped, must leave every paper setting at its default
    and every optional feature off, and the run cell must call run_experiment(cfg).
    """
    import json
    import re
    _stub_missing_modules()
    nb = json.load(open(os.path.join(REPO, "LASSI_pipeline.ipynb")))
    code = [re.sub(r"^\s*%.*$", "", "".join(c["source"]), flags=re.M)
            for c in nb["cells"] if c["cell_type"] == "code"]
    assert code[-1].strip() == "run_experiment(cfg)", "last notebook cell must run the pipeline"

    old_cwd, old_path = os.getcwd(), list(sys.path)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    try:
        g = {}
        for src in code[:-1]:
            exec(src, g)
        cfg, defaults = g["cfg"], g["Config"]()
        # Fields a user is expected to set for their machine; everything else must stay default.
        machine_specific = {"app", "source_lang", "target_lang", "llm_source", "model", "num_ctx"}
        for name in vars(defaults):
            if name not in machine_specific:
                assert getattr(cfg, name) == getattr(defaults, name), (
                    f"notebook changes the default of {name!r}: {getattr(cfg, name)!r}")
        for name, off in OPTIONAL_OFF.items():
            assert getattr(cfg, name) == off, f"optional feature {name!r} is on in the shipped notebook"
    finally:
        os.chdir(old_cwd); sys.path[:] = old_path


if __name__ == "__main__":
    test_cli_matches_original()
    test_cli_options_and_errors()
    test_optional_features()
    test_notebook_ships_paper_defaults()
    test_prompt_dictionary_matches_original()
    test_default_config_matches_original()
    print("PASS: current LASSI (default config) sends exactly the same prompts as the original.")
