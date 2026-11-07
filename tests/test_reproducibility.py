"""
Reproducibility test: with its default configuration, the current LASSI notebook must behave
exactly like the original notebook that produced the paper's results.

Both notebooks are run on the same small kernel with a scripted LLM that forces one compile
error and one execution error, so initial, compile-correction and execution-correction
prompts are all exercised. Every prompt sent to the LLM must be identical, and so must the
final code that is written. Requires only g++ (no GPU, model or network).

Run:  python tests/test_reproducibility.py      (or: pytest tests)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, run_notebook

LEGACY = os.path.join(REPO, "tests", "legacy", "LASSI_pipeline_v0_original.ipynb")
CURRENT = os.path.join(REPO, "LASSI_pipeline_v0.ipynb")


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


def test_default_config_matches_original():
    assert shutil.which("g++"), "g++ is required for this test"
    legacy = run_notebook(LEGACY)
    current = run_notebook(CURRENT)

    for name in PAPER_DEFAULTS:
        assert legacy["defaults"][name] == current["defaults"][name], (
            f"default {name!r} changed: original {legacy['defaults'][name]!r}, current {current['defaults'][name]!r}")
    for name, off in OPTIONAL_OFF.items():
        assert current["defaults"][name] == off, f"optional feature {name!r} is not off by default"

    assert legacy["completed"] and current["completed"], "pipeline did not complete"
    assert len(legacy["transcript"]) == 5, (   # summary, description, initial, 2 corrections
        f"unexpected scenario: {len(legacy['transcript'])} LLM calls in the original")
    assert len(current["transcript"]) == len(legacy["transcript"]), (
        f"LLM call count differs: original {len(legacy['transcript'])}, current {len(current['transcript'])}")

    i, a, b = _first_difference(legacy["transcript"], current["transcript"])
    assert a is None, f"prompt #{i} differs between original and current:\n--- original\n{a}\n--- current\n{b}"

    assert list(legacy["files"].values()) == list(current["files"].values()), "generated code differs"


if __name__ == "__main__":
    test_default_config_matches_original()
    print("PASS: current notebook (default config) sends exactly the same prompts as the original.")
