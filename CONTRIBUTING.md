# Contributing

LASSI is research code accompanying a published paper (see [`CITATION.cff`](CITATION.cff)). With no optional flags the pipeline, prompts and defaults reproduce that paper, so changes that affect reproducibility will be weighed against it. `python tests/test_reproducibility.py` checks that the defaults still match the original; please run it before sending a change to the pipeline or the prompts.

Bug reports and small fixes are welcome via GitHub issues and pull requests. For larger changes or new features, please open an issue first to discuss the approach before submitting a PR. This is a research project, so reviews are best-effort.

When reporting a problem, please include your operating system, your compilers (`nvcc` / `nvc++` or `g++`) and GPU with driver version if you have one, the LLM backend and model, the command you ran, and the relevant part of its output. Never include API keys, credentials or the contents of your `config.yaml` in an issue or PR.

New HeCBench applications can be added without code changes; see [Adding an Application](README.md#adding-an-application).
