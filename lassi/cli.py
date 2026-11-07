"""
Command-line interface for LASSI.

    python run_lassi.py --app jacobi --source CUDA --target OMP --llm-source ollama --model <name>

With no optional flags this runs the pipeline of the original LASSI paper. It builds the same
Config and calls the same run_experiment() as the notebook, so the two never diverge.
"""
import argparse
import shlex
import sys

from . import llm_backends
from .config import Config, load_apps
from .pipeline import run_experiment


def build_parser():
    defaults = Config()
    p = argparse.ArgumentParser(
        prog="run_lassi.py",
        description="LASSI: translate a HeCBench application between CUDA and OpenMP (GPU offload) with an LLM, "
                    "self-correcting from compiler and runtime errors. Defaults reproduce the original paper.",
        epilog="LLM connection settings (URLs, API key, Argo username) normally live in config.yaml (copy "
               "config.yaml.example); the flags below override it. Environment variables OPENAI_API_KEY, "
               "OPENAI_BASE_URL, LASSI_ARGO_URL, LASSI_ARGO_USERNAME and OLLAMA_HOST are used when config.yaml "
               "does not set the value.")

    g = p.add_argument_group("what to translate")
    g.add_argument("--app", default=defaults.app, help="HeCBench application folder name (default: %(default)s)")
    g.add_argument("--source", dest="source_lang", default=defaults.source_lang, choices=["CUDA", "OMP"],
                   help="source language (default: %(default)s)")
    g.add_argument("--target", dest="target_lang", default=defaults.target_lang, choices=["CUDA", "OMP"],
                   help="target language (default: %(default)s)")
    g.add_argument("--exe-params", default=None,
                   help="command-line arguments for the program as one quoted string, e.g. \"10000 8 1\" "
                        "(default: from source_apps.json)")
    g.add_argument("--list-apps", action="store_true", help="list the available applications and exit")

    g = p.add_argument_group("LLM")
    g.add_argument("--llm-source", default=defaults.llm_source, choices=list(llm_backends.LLM_SOURCES),
                   help="LLM backend (default: %(default)s)")
    g.add_argument("--model", default=defaults.model, help="model name (default: %(default)s)")
    g.add_argument("--num-ctx", type=int, default=defaults.num_ctx,
                   help="context window in tokens, ollama only (default: %(default)s)")
    g.add_argument("--config", default=None, metavar="FILE",
                   help="connection settings file (default: ./config.yaml; see config.yaml.example)")
    g.add_argument("--ollama-host", default=None, help="Ollama server URL (default: local server)")
    g.add_argument("--lmstudio-url", default=None, help="LM Studio server URL (default: http://localhost:1234/v1)")
    g.add_argument("--openai-url", default=None,
                   help="OpenAI-compatible endpoint base URL (default: https://api.openai.com/v1)")
    g.add_argument("--openai-api-key", default=None,
                   help="API key for the OpenAI-compatible endpoint (better kept in config.yaml than typed here, "
                        "where it lands in your shell history)")
    g.add_argument("--argo-url", default=None, help="Argo Gateway streamchat URL (Argonne network only)")
    g.add_argument("--argo-username", default=None,
                   help="your Argonne domain username, used as the Argo key (not a secret)")
    g.add_argument("--argo-ca-bundle", default=None,
                   help="path to a CA certificate bundle for the Argo gateway's TLS certificate")

    g = p.add_argument_group("compiling and running")
    g.add_argument("--compiler", default=None, help="compiler for the target language (default: the paper's)")
    g.add_argument("--compiler-flags", default=None, help="compiler flags as one quoted string (default: the paper's)")
    g.add_argument("--no-execute", dest="yes_execute", action="store_false",
                   help="only compile the generated code; do not execute it")
    g.add_argument("--cpu", dest="cpu_only", action="store_true",
                   help="compile OpenMP with g++ -fopenmp so everything runs on a CPU, no GPU or NVIDIA compiler "
                        "(CUDA -> OMP only; --compiler/--compiler-flags still override)")
    g.add_argument("--run-timeout", type=float, default=None, metavar="SECONDS",
                   help="stop any program that runs longer than this (default: no limit)")

    g = p.add_argument_group("pipeline")
    g.add_argument("--no-context", dest="use_context", action="store_false",
                   help="skip the language-knowledge context and self-prompting")
    g.add_argument("--experiment", default=defaults.experiment, help="label used in output file names (default: %(default)s)")
    g.add_argument("--quiet", dest="display_response", action="store_false", help="do not echo LLM responses")

    g = p.add_argument_group("optional extras (all off by default)")
    g.add_argument("--judge", dest="use_judge", action="store_true",
                   help="ask an LLM whether the final translation is functionally equivalent to the reference")
    g.add_argument("--judge-model", default=None, help="judge model on the same backend (default: same as --model)")
    g.add_argument("--record-results", action="store_true", help="append one row per trial to a results CSV")
    g.add_argument("--trials", type=int, default=defaults.total_trials, help="number of trials to run (default: %(default)s)")
    g.add_argument("--timing-repeats", type=int, default=defaults.timing_repeats, metavar="N",
                   help="time the reference and the translation N times each and report the mean; the paper "
                        "used 3 (default: %(default)s)")
    g.add_argument("--max-self-corrections", type=int, default=None, metavar="N",
                   help="give up on a trial after N self-corrections (default: never give up, as in the paper)")
    return p


def config_from_args(args):
    """Turn parsed arguments into a Config (applies the backend connection settings as a side effect)."""
    if args.config:
        llm_backends.load_config_file(args.config)
    # Only flags that were actually given override config.yaml / the environment
    given = {"ollama_host": args.ollama_host, "lmstudio_base_url": args.lmstudio_url,
             "openai_base_url": args.openai_url, "openai_api_key": args.openai_api_key,
             "argo_url": args.argo_url, "argo_username": args.argo_username,
             "argo_verify": args.argo_ca_bundle}
    llm_backends.configure(**{name: value for name, value in given.items() if value})
    return Config(
        app=args.app, source_lang=args.source_lang, target_lang=args.target_lang,
        exe_params=shlex.split(args.exe_params) if args.exe_params is not None else None,
        llm_source=args.llm_source, model=args.model, num_ctx=args.num_ctx,
        use_context=args.use_context, yes_execute=args.yes_execute,
        compiler=args.compiler, compiler_flags=args.compiler_flags,
        experiment=args.experiment, display_response=args.display_response,
        use_judge=args.use_judge, model_judge=args.judge_model,
        record_results=args.record_results, total_trials=args.trials,
        timing_repeats=args.timing_repeats, max_self_corrections=args.max_self_corrections,
        run_timeout_s=args.run_timeout, cpu_only=args.cpu_only,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_apps:
        for name, info in sorted(load_apps().items()):
            print(f"{name:18s} params: {' '.join(info['app_params']) or '(none)':26s} {info['code_lines']} lines")
        return 0

    try:
        cfg = config_from_args(args)
        if cfg.total_trials < 1:
            parser.error("--trials must be at least 1")
        outcomes = run_experiment(cfg)
    except (ValueError, FileNotFoundError) as e:
        parser.exit(2, f"error: {e}\n")
    return 0 if outcomes and all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
