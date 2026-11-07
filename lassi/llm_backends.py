"""
LLM inference backends for the LASSI pipeline.

One entry point, ``call_llm()``, dispatches to one of four backends selected by the
``llm_source`` string:

    "ollama"             Local Ollama server (https://ollama.com)
    "lmstudio"           Local LM Studio server (OpenAI-compatible, default port 1234)
    "openai_compatible"  Any server that implements the OpenAI chat-completions REST
                         contract: OpenAI itself, vLLM, llama.cpp server, OpenRouter,
                         an institutional gateway, ...
    "argo"               Argonne National Laboratory's Argo Gateway (Argonne network
                         access only). Uses the gateway's native "streamchat" API.

Nothing in this module is hardcoded to a particular host, user or key. Connection
details are set with ``configure()`` (or by assigning the module globals below), and
secrets are read from environment variables when they are left blank:

    OPENAI_API_KEY   used by "openai_compatible" when ``openai_api_key`` is empty

Example::

    import llm_backends as llm
    llm.configure(openai_base_url="https://api.openai.com/v1")      # key from $OPENAI_API_KEY
    text = llm.call_llm("openai_compatible", "gpt-4.1-mini", 0, "You are helpful.", "Hello")

Argo Gateway note: an Argo "API key" is just your Argonne domain username, so it is
not a secret. The same gateway also exposes an OpenAI-compatible endpoint; you can use
that through the "openai_compatible" backend (set ``openai_base_url`` to the gateway's
OpenAI-compatible URL and ``openai_api_key`` to your username).
"""

import json
import os
import re
import subprocess

import requests

# ----------------------------------------------------------------------------
# Connection settings -- set via configure() or by assigning these globals.
# ----------------------------------------------------------------------------

# Ollama: None means the ollama client's default (http://localhost:11434, or $OLLAMA_HOST).
ollama_host = None

# LM Studio's local OpenAI-compatible server.
lmstudio_base_url = "http://localhost:1234/v1"

# Generic OpenAI-compatible backend. If openai_api_key is blank, $OPENAI_API_KEY is used.
openai_base_url = "https://api.openai.com/v1"
openai_api_key = ""

# Argo Gateway (native streamchat API). Both must be provided by the user:
#   argo_url      full URL of the streamchat resource, i.e. "https://<gateway-host>/argoapi/api/v1/resource/streamchat/"
#   argo_username your Argonne domain username (identifies you to the gateway; not a secret)
# argo_verify controls TLS certificate checking: True (default), False, or a path to a CA bundle.
argo_url = ""
argo_username = ""
argo_verify = True

# Seconds to wait on a single LLM response before giving up.
request_timeout = 600

# Print streamed responses (Argo) to the console as they arrive. LASSI sets this from its
# display_response setting, so --quiet silences the stream as well.
echo_stream = True

_SETTINGS = ("ollama_host", "lmstudio_base_url", "openai_base_url", "openai_api_key",
             "argo_url", "argo_username", "argo_verify", "request_timeout", "echo_stream")

LLM_SOURCES = ("ollama", "lmstudio", "openai_compatible", "argo")


def configure(**kwargs):
    """Set one or more connection settings, e.g. configure(argo_url=..., argo_username=...)."""
    for name, value in kwargs.items():
        if name not in _SETTINGS:
            raise ValueError(f"Unknown setting {name!r}; valid settings: {', '.join(_SETTINGS)}")
        globals()[name] = value


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def _show_prompt(system_prompt, content_prompt, temp, verifyprompt):
    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")


def _strip_think(text):
    """Remove <think>...</think> reasoning blocks that some models prepend to answers."""
    if re.search(r"<think>.*?</think>", text, flags=re.DOTALL):
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text.strip()


def _bearer_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ----------------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------------

def _ollama_client():
    try:
        import ollama
    except ImportError as e:
        raise ImportError("The 'ollama' backend needs the ollama package: pip install ollama") from e
    return ollama.Client(host=ollama_host) if ollama_host else ollama.Client()


def ollama_llm(model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    """
    Local Ollama server. Requires `ollama serve` to be running and the model to have been
    pulled (`ollama pull <model>`). `num_ctx` sets the context window in tokens.
    """
    _show_prompt(system_prompt, content_prompt, temp, verifyprompt)
    response = _ollama_client().chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt},
        ],
        stream=False,
        options=dict(temperature=temp, top_p=0.9, num_ctx=num_ctx),
    )
    return response["message"]["content"]


def unload_ollama_model(model):
    """Ask the Ollama server to free the model's memory (keep_alive=0)."""
    print("\nUnloading Ollama model...")
    try:
        _ollama_client().generate(model=model, keep_alive=0)
    except Exception as e:  # unloading is best-effort; never stop a run over it
        print(f"(could not unload {model!r}: {e})")


def list_ollama_models():
    """Return the names of the models available on the Ollama server."""
    return sorted(m["model"] for m in _ollama_client().list()["models"])


# ----------------------------------------------------------------------------
# OpenAI-compatible chat completions (also used for LM Studio)
# ----------------------------------------------------------------------------

# Sampling parameters sent on every request. Some models and servers reject one of them --
# OpenAI's reasoning models reject both. The first rejection is detected from the 400
# response, the request is retried without that parameter, and the restriction is
# remembered per model for the rest of the process, so the discovery cost is paid once.
_SAMPLING_PARAMS = ("temperature", "top_p")
_unsupported_sampling_params = {}  # model name -> set of rejected parameter names


def _rejected_sampling_param(resp):
    """
    If `resp` is a 400 saying one of our sampling parameters is unsupported for this model,
    return that parameter's name, else None. Understands OpenAI's JSON error object and
    gateways that forward the upstream error as a Python-repr string.
    """
    if resp.status_code != 400:
        return None
    text = resp.text

    try:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        if (isinstance(err, dict) and err.get("param") in _SAMPLING_PARAMS
                and err.get("code") in (None, "unsupported_value", "unsupported_parameter")):
            return err["param"]
    except ValueError:
        pass

    if "unsupported" in text.lower():
        m = re.search(r"""param['"]?\s*:\s*['"](temperature|top_p)['"]""", text)
        if m:
            return m.group(1)
        m = re.search(r"""unsupported[^.\n]{0,40}['"](temperature|top_p)['"]""", text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def _chat_completions(base_url, api_key, model, system_prompt, content_prompt, temp, send_top_p=True):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt},
        ],
    }
    unsupported = _unsupported_sampling_params.setdefault(model, set())
    if "temperature" not in unsupported:
        payload["temperature"] = round(temp, 1)
    if send_top_p and "top_p" not in unsupported:
        payload["top_p"] = 0.9

    url = base_url.rstrip("/") + "/chat/completions"
    headers = _bearer_headers(api_key)
    resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)

    # At most one retry per sampling parameter: drop whichever one the model rejected.
    for _ in range(len(_SAMPLING_PARAMS)):
        rejected = _rejected_sampling_param(resp)
        if rejected is None or rejected not in payload:
            break
        print(f"\n[llm_backends] Model {model!r} does not accept {rejected}={payload[rejected]} "
              f"-- retrying without it (and skipping it for later {model!r} calls).")
        unsupported.add(rejected)
        payload.pop(rejected)
        resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.HTTPError(f"{e} -- response body: {resp.text}", response=resp) from None

    return _strip_think(resp.json()["choices"][0]["message"]["content"])


def openai_compatible_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    """Any OpenAI-compatible server. Uses openai_base_url and openai_api_key / $OPENAI_API_KEY."""
    _show_prompt(system_prompt, content_prompt, temp, verifyprompt)
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    return _chat_completions(openai_base_url, api_key, model, system_prompt, content_prompt, temp)


def lmstudio_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    """Local LM Studio server (start its local server in the app). No real key is needed."""
    _show_prompt(system_prompt, content_prompt, temp, verifyprompt)
    # The original LASSI sent only temperature to LM Studio (no top_p); keep it that way.
    return _chat_completions(lmstudio_base_url, "lm-studio", model, system_prompt, content_prompt, temp, send_top_p=False)


def list_openai_compatible_models(base_url=None, api_key=None):
    """
    Query {base_url}/models and return a sorted list of (usable_model_id, display_label).

    Argo Gateway's OpenAI-compatible endpoint puts a human-readable name in the standard
    "id" field and the machine-usable slug in a non-standard "internal_id" field; passing
    "id" as the chat-completions model is rejected. Plain servers have no "internal_id",
    so "id" is the usable value there and both tuple entries are the same.
    """
    base_url = base_url or openai_base_url
    api_key = api_key if api_key is not None else (openai_api_key or os.environ.get("OPENAI_API_KEY", ""))
    resp = requests.get(base_url.rstrip("/") + "/models", headers=_bearer_headers(api_key), timeout=15)
    resp.raise_for_status()
    return sorted(((m.get("internal_id") or m["id"]), m["id"]) for m in resp.json()["data"])


# ----------------------------------------------------------------------------
# Argo Gateway (native streamchat API)
# ----------------------------------------------------------------------------

def argo_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    """
    Argo Gateway native "streamchat" API. Reachable only from the Argonne network (or a
    tunnel into it). Requires configure(argo_url=..., argo_username=...). The response is
    streamed and echoed to the console as it arrives.
    """
    if not argo_url or not argo_username:
        raise ValueError("The 'argo' backend needs argo_url and argo_username -- call "
                         "llm_backends.configure(argo_url=..., argo_username=...).")

    # Some reasoning models do not accept a separate system message.
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt

    _show_prompt(system_prompt, content_prompt, temp, verifyprompt)

    data = {
        "user": argo_username,
        "model": model,
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop": [],
        "temperature": round(temp, 1),
    }
    # Claude models do not accept top_p alongside temperature.
    if not model.lower().startswith("claude"):
        data["top_p"] = 0.9

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/plain",
        "Accept-Encoding": "identity",  # no gzip, so chunks stream as they are produced
        "Cache-Control": "no-cache",
    }

    if argo_verify is False:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    response_text = ""
    with requests.post(argo_url, headers=headers, json=data, stream=True,
                       verify=argo_verify, timeout=request_timeout) as resp:
        if resp.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Argo Gateway returned HTTP {resp.status_code}: {resp.text}", response=resp)
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                if echo_stream:
                    print(chunk, end="", flush=True)
                response_text += chunk
    if echo_stream:
        print()

    return _strip_think(response_text)


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

def call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    """Single dispatch point for every backend. `num_ctx` is only used by "ollama"."""
    if llm_source == "ollama":
        return ollama_llm(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "lmstudio":
        return lmstudio_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "openai_compatible":
        return openai_compatible_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "argo":
        return argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    raise ValueError(f"Unknown llm_source {llm_source!r}; expected one of {LLM_SOURCES}")
