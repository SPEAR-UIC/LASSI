# Security Policy

LASSI is research software. It compiles and executes LLM-generated code on your machine and sends source code to the LLM backend you configure, so run it only in an environment where that is acceptable (ideally a dedicated or sandboxed machine or GPU node). Use `--run-timeout` to stop programs that run too long.

## Reporting a vulnerability

Please report security issues privately rather than in a public issue. Use GitHub's **Report a vulnerability** option on the repository's Security tab, or email mdear2@uic.edu. Reports are handled on a best-effort basis.

## Credentials

LASSI reads LLM connection settings and API keys from `config.yaml` (which is listed in `.gitignore`), from command-line flags, or from environment variables such as `OPENAI_API_KEY`. Do not commit keys, and keep `config.yaml` private. Prefer `config.yaml` or an environment variable over `--openai-api-key`, which lands in your shell history. Clear notebook outputs before sharing a notebook. If you find a credential in this repository, please report it so it can be revoked.
