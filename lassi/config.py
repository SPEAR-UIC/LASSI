"""
LASSI settings.

Every default below reproduces the original LASSI paper (Dearing et al., LLMxHPC @ CLUSTER 2024):
context knowledge on, self-prompting, self-correction until the code compiles and runs, the
paper's prompts and compile flags. Features added after the paper (LLM judge, results CSV,
repeated trials) are off unless switched on.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional

_APPS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source_apps.json")

# Compilers and flags the paper used on an NVIDIA A100 (compute capability 8.0).
# Adjust `compiler` / `compiler_flags` (and the -arch / -gpu flags) for other GPUs.
_PAPER_COMPILERS = {
    "CUDA": ("nvcc", "-std=c++14 -Xcompiler -Wall -arch=sm_80 -O3"),
    "OMP":  ("nvc++", "-Wall -O3 -Minfo -mp=gpu -gpu=cc80"),
}

_EXTENSIONS = {"CUDA": "cu", "HIP": "cu", "OMP": "cpp", "SYCL": "cpp"}


def load_apps(path=_APPS_FILE):
    """The benchmark registry: {app name: {"app_folder", "app_params", "code_lines"}}."""
    with open(path) as f:
        apps = json.load(f)["HeCBench"]
    return {k: v for k, v in apps.items() if isinstance(v, dict)}


@dataclass
class Config:
    # --- what to translate -------------------------------------------------------------
    app: str = "jacobi"                   # a HeCBench application folder name (see source_apps.json)
    source_lang: str = "CUDA"             # CUDA | OMP
    target_lang: str = "OMP"              # OMP | CUDA
    exe_params: Optional[List[str]] = None  # command-line arguments for the program; None = from source_apps.json

    # --- the LLM -----------------------------------------------------------------------
    llm_source: str = "ollama"            # ollama | lmstudio | openai_compatible | argo
    model: str = "wizardcoder:33b-v1.1-q8_0"
    num_ctx: int = 16384                  # context window in tokens (used by ollama only)

    # --- the paper's pipeline ----------------------------------------------------------
    use_context: bool = True              # language-knowledge context + self-prompting
    yes_execute: bool = True              # execute (not just compile) the generated code
    compiler: Optional[str] = None        # None = the paper's compiler for the target language
    compiler_flags: Optional[str] = None  # None = the paper's flags
    experiment: str = "01a_withcontext"   # label used in output file names
    display_response: bool = True         # show LLM responses as they arrive
    test_with_error: bool = False
    generate_test_case: bool = False

    # --- optional features (off by default) --------------------------------------------
    use_judge: bool = False               # LLM-as-a-judge verdict on the final translation
    model_judge: Optional[str] = None     # judge model on the same llm_source; None = `model`
    record_results: bool = False          # append a row per trial to a results CSV
    total_trials: int = 1                 # repeat the whole pipeline (e.g., for pass@k)

    # ---------------------------------------------------------------------------------
    def __post_init__(self):
        for name, value in (("source_lang", self.source_lang), ("target_lang", self.target_lang)):
            if value not in _EXTENSIONS:
                raise ValueError(f"{name} must be one of {sorted(_EXTENSIONS)}, got {value!r}")
        if self.source_lang == self.target_lang:
            raise ValueError("source_lang and target_lang must differ")
        if self.model_judge is None:
            self.model_judge = self.model

    @property
    def context_knowledge(self):
        """Key into the prompt dictionary's language-knowledge entries (the target language)."""
        return "cuda" if self.target_lang == "CUDA" else self.target_lang.lower()

    def resolved_compiler(self):
        """(compiler, flags): the user's choice, else the paper's for the target language."""
        default = _PAPER_COMPILERS.get(self.target_lang, ("", ""))
        return (self.compiler if self.compiler is not None else default[0],
                self.compiler_flags if self.compiler_flags is not None else default[1])

    def resolved_exe_params(self):
        if not self.yes_execute:
            return [""]
        if self.exe_params is not None:
            return list(self.exe_params)
        apps = load_apps()
        if self.app not in apps:
            raise ValueError(f"Unknown app {self.app!r}; known apps: {sorted(apps)}. "
                             "Pass exe_params explicitly or add it to source_apps.json.")
        return list(apps[self.app]["app_params"])

    @staticmethod
    def extension(lang):
        return _EXTENSIONS[lang]
