"""LASSI: an LLM-based automated self-correcting pipeline for translating parallel scientific codes."""
from . import llm_backends
from .config import Config, load_apps
from .pipeline import prepare_experiment, run_experiment

# Pick up connection settings (LLM URLs, keys, usernames) from config.yaml, if there is one.
llm_backends.load_config_file()

__all__ = ["Config", "load_apps", "prepare_experiment", "run_experiment"]
