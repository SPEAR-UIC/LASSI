"""
LLM backend tests against small local fake servers (no network, no real model needed).

Covers the OpenAI-compatible backend (bearer auth, <think> stripping, dropping a sampling
parameter a model rejects and remembering it), LM Studio (temperature only, as in the original
LASSI), Argo streaming (user field, no top_p for Claude models, console echo on/off),
model listing, and configuration errors.

Run:  python tests/test_backends.py
"""
import contextlib
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, _stub_missing_modules

_stub_missing_modules()
sys.path.insert(0, REPO)
from lassi import llm_backends as llm  # noqa: E402


class _FakeServer:
    """Speaks just enough OpenAI chat-completions, /models and Argo streamchat to test against."""

    def __init__(self):
        self.calls = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": "GPT Nice", "internal_id": "gptnice"}, {"id": "plain"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.endswith("/chat/completions"):
                    outer.calls.append({"kind": "chat", "model": body["model"], "auth": self.headers.get("Authorization"),
                                        "sampling": sorted(k for k in body if k in ("temperature", "top_p"))})
                    if body["model"] == "reasoner" and "temperature" in body:      # like OpenAI's reasoning models
                        return self._send(400, {"error": {"param": "temperature", "code": "unsupported_value",
                                                          "message": "only the default is supported"}})
                    reply = "<think>hmm</think>ANSWER from " + body["model"]
                    return self._send(200, {"choices": [{"message": {"content": reply}}]})
                # Argo streamchat: chunked plain text
                outer.calls.append({"kind": "argo", "user": body["user"], "model": body["model"],
                                    "has_top_p": "top_p" in body})
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for part in ("Hel", "lo ", "Argo"):
                    self.wfile.write(f"{len(part):x}\r\n{part}\r\n".encode())
                    self.wfile.flush()
                    time.sleep(0.02)
                self.wfile.write(b"0\r\n\r\n")

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@contextlib.contextmanager
def _fake():
    server = _FakeServer()
    saved = {k: getattr(llm, k) for k in llm._SETTINGS}
    saved_unsupported = dict(llm._unsupported_sampling_params)
    llm._unsupported_sampling_params.clear()
    llm.configure(openai_base_url=server.base, openai_api_key="KEY123", lmstudio_base_url=server.base,
                  argo_url=server.base + "/argoapi/streamchat/", argo_username="someuser")
    try:
        yield server
    finally:
        llm.configure(**saved)
        llm._unsupported_sampling_params.clear()
        llm._unsupported_sampling_params.update(saved_unsupported)
        server.close()


def _quiet_call(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        return llm.call_llm(*args, **kwargs), out.getvalue()


def test_openai_compatible():
    with _fake() as s:
        reply, _ = _quiet_call("openai_compatible", "gpt-x", 0, "sys", "hi")
        assert reply == "ANSWER from gpt-x", reply                      # <think> block stripped
        assert s.calls[0]["auth"] == "Bearer KEY123"
        assert s.calls[0]["sampling"] == ["temperature", "top_p"]


def test_rejected_sampling_parameter_is_dropped_and_remembered():
    with _fake() as s:
        reply, _ = _quiet_call("openai_compatible", "reasoner", 0, "sys", "hi")
        assert reply == "ANSWER from reasoner"
        reasoner = [c["sampling"] for c in s.calls if c["model"] == "reasoner"]
        assert reasoner == [["temperature", "top_p"], ["top_p"]], reasoner   # one rejection, then a retry
        _quiet_call("openai_compatible", "reasoner", 0, "sys", "again")
        assert [c["sampling"] for c in s.calls if c["model"] == "reasoner"][-1] == ["top_p"]
        assert len([c for c in s.calls if c["model"] == "reasoner"]) == 3    # no second rejection round trip


def test_lmstudio_sends_temperature_only():
    """The original LASSI sent only temperature to LM Studio; that must stay true."""
    with _fake() as s:
        _quiet_call("lmstudio", "m", 0, "sys", "hi")
        assert s.calls[0]["sampling"] == ["temperature"] and s.calls[0]["auth"] == "Bearer lm-studio"


def test_argo_streaming():
    with _fake() as s:
        reply, shown = _quiet_call("argo", "gpt4o", 0, "sys", "hi")
        assert reply == "Hello Argo" and "Hello Argo" in shown
        assert s.calls[0] == {"kind": "argo", "user": "someuser", "model": "gpt4o", "has_top_p": True}
        _quiet_call("argo", "claudesonnet4", 0, "sys", "hi")
        assert s.calls[1]["has_top_p"] is False                              # Claude models take no top_p
        llm.configure(echo_stream=False)
        reply, shown = _quiet_call("argo", "gpt4o", 0, "sys", "hi")
        assert reply == "Hello Argo" and "Hello" not in shown.replace("LLM Prompt Not Shown", "")


def test_list_models_uses_internal_id():
    with _fake():
        assert llm.list_openai_compatible_models() == [("gptnice", "GPT Nice"), ("plain", "plain")]


def test_configuration_errors():
    with _fake():
        llm.configure(argo_url="", argo_username="")
        for call, text in ((lambda: llm.call_llm("argo", "m", 0, "s", "p"), "argo_url"),
                           (lambda: llm.call_llm("nope", "m", 0, "s", "p"), "Unknown llm_source"),
                           (lambda: llm.configure(bogus=1), "Unknown setting")):
            try:
                call()
                raise AssertionError(f"expected an error mentioning {text!r}")
            except ValueError as e:
                assert text in str(e), e


def test_no_hardcoded_hosts_or_users():
    """The backend module must not contain a gateway host or a person's username."""
    source = open(os.path.join(REPO, "lassi", "llm_backends.py")).read().lower()
    for forbidden in ("anl.gov", "mdearing", "svclassi", "localhost:5000", "apps-dev"):
        assert forbidden not in source, f"{forbidden!r} is hardcoded in llm_backends.py"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS  {name}")
