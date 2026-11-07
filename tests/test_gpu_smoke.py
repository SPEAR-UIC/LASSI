"""
GPU smoke tests: the pipeline's compile / run / time path with the REAL compilers on a REAL GPU.

The LLM is still scripted (it hands back the reference implementation), so these tests are
fast, free and deterministic; what they prove is that nvcc, nvc++ (OpenMP GPU offload), the
runner and the records work together on this machine, in both translation directions.

Skipped automatically when nvidia-smi, nvcc or nvc++ is missing.

Run:  python tests/test_gpu_smoke.py
"""
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, _stub_missing_modules, run_current

APP = "layout"          # runs in well under a second on a GPU
APP_ARGS = ("1",)
SRC = os.path.join(REPO, "translated_code", "input_codes", "HeCBench", APP)


def _require_gpu():
    """Return the GPU's compute capability as digits, e.g. '89'; skip the test if there is no usable GPU."""
    for tool in ("nvidia-smi", "nvcc", "nvc++"):
        if not shutil.which(tool):
            raise unittest.SkipTest(f"{tool} not found")
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise unittest.SkipTest("no NVIDIA GPU reported by nvidia-smi")
    return out.stdout.strip().splitlines()[0].replace(".", "")


def _paper_flags(cc):
    """The paper's compile flags (A100 = compute capability 8.0) retargeted at this GPU."""
    _stub_missing_modules()
    sys.path.insert(0, REPO)
    from lassi import Config
    flags = {lang: Config(source_lang=src, target_lang=lang).resolved_compiler()
             for lang, src in (("CUDA", "OMP"), ("OMP", "CUDA"))}
    assert "sm_80" in flags["CUDA"][1] and "cc80" in flags["OMP"][1], "paper flags changed"
    return {lang: (cmd, f.replace("sm_80", f"sm_{cc}").replace("cc80", f"cc{cc}"))
            for lang, (cmd, f) in flags.items()}


def test_reference_programs_run_on_gpu():
    """The CUDA reference builds with nvcc; the OpenMP reference offloads to the GPU with nvc++."""
    cc = _require_gpu()
    _stub_missing_modules()
    sys.path.insert(0, REPO)
    from lassi.runner import compile_code, execute_code
    flags = _paper_flags(cc)

    with tempfile.TemporaryDirectory(prefix="lassi_gpu_") as d:
        for name in os.listdir(SRC):
            if not name.endswith("_main"):
                shutil.copy(os.path.join(SRC, name), d)

        # CUDA, built the way the pipeline builds it
        err, exe = compile_code(os.path.join(d, f"{APP}-cuda_main.cu"), "CUDA", *flags["CUDA"], include_dir=d)
        assert err == "", f"nvcc failed:\n{err}"
        result, out, runtime = execute_code(exe, list(APP_ARGS), echo=False, timeout=120)
        assert result == "0" and out.strip() and runtime > 0, (result, out, runtime)

        # OpenMP: confirm the compiler really generated GPU kernels, then run it
        cmd = [flags["OMP"][0], *flags["OMP"][1].split(), os.path.join(d, f"{APP}-omp_main.cpp"),
               "-o", os.path.join(d, "layout_omp")]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        assert "GPU kernel" in build.stderr, "nvc++ did not offload any region to the GPU:\n" + build.stderr[:600]
        result, out, runtime = execute_code(os.path.join(d, "layout_omp"), list(APP_ARGS), echo=False, timeout=120)
        assert result == "0" and out.strip() and runtime > 0, (result, out, runtime)


def _run_direction(source, target):
    cc = _require_gpu()
    compiler, flags = _paper_flags(cc)[target]
    r = run_current(app=APP, app_args=APP_ARGS, scenario="clean", source_lang=source, target_lang=target,
                    compiler=compiler, compiler_flags=flags, record_results=True, timing_repeats=2)
    assert r["completed"], "the pipeline did not complete"
    assert len(r["transcript"]) == 3        # context summary, code description, one generation: no self-correction needed
    (csv_text,) = r["csv"].values()
    (row,) = list(csv.DictReader(io.StringIO(csv_text)))
    assert row["lassi_outcome"] == "success" and row["lassi_compiled"] == "Y" and row["lassi_execute"] == "Y", row
    assert row["source_language"] == source and row["target_language"] == target, row
    assert float(row["target_orig_runtime_s"]) > 0 and float(row["lassi_runtime_s"]) > 0, row
    assert row["timing_repeats"] == "2" and row["lassi_runtime_stdev_s"] != "", row
    assert float(row["lassi_execute_ratio"]) > 0, row


def test_pipeline_cuda_to_openmp_on_gpu():
    _run_direction("CUDA", "OMP")


def test_pipeline_openmp_to_cuda_on_gpu():
    _run_direction("OMP", "CUDA")


if __name__ == "__main__":
    for test in (test_reference_programs_run_on_gpu, test_pipeline_cuda_to_openmp_on_gpu,
                 test_pipeline_openmp_to_cuda_on_gpu):
        try:
            test()
            print(f"PASS  {test.__name__}")
        except unittest.SkipTest as e:
            print(f"SKIP  {test.__name__}: {e}")
