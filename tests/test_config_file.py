"""
Tests for config.yaml: where the LLM connection settings (URLs, API key, Argo username) live.

Priority, highest first: command-line flag / configure(), then config.yaml, then an environment
variable, then the built-in default.

Run:  python tests/test_config_file.py
"""
import importlib
import os
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, _stub_missing_modules

_stub_missing_modules()
sys.path.insert(0, REPO)
import lassi  # noqa: E402
from lassi import llm_backends  # noqa: E402

_ENV_VARS = ("OLLAMA_HOST", "OPENAI_BASE_URL", "OPENAI_API_KEY", "LASSI_ARGO_URL", "LASSI_ARGO_USERNAME")
_FULL = """
    ollama:
      host: "http://ollama.example:11434"
    lmstudio:
      base_url: "http://lmstudio.example:1234/v1"
    openai_compatible:
      base_url: "https://llm.example/v1"
      api_key: "sk-from-the-file"
    argo:
      url: "https://argo.example/streamchat/"
      username: "someuser"
      ca_bundle: "/path/to/ca.pem"
"""


def _settings():
    return {name: getattr(llm_backends, name) for name in llm_backends._SETTINGS}


class _Fresh:
    """A pristine llm_backends (module defaults re-read from a controlled environment), restored afterwards."""

    def __init__(self, **env):
        self.env = env

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in (*_ENV_VARS, "LASSI_CONFIG")}
        for k in _ENV_VARS:
            os.environ.pop(k, None)
        os.environ["LASSI_CONFIG"] = ""
        os.environ.update(self.env)
        importlib.reload(llm_backends)
        return llm_backends

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(llm_backends)


def _write(directory, text, name="config.yaml"):
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))
    return path


def test_file_values_are_applied():
    with _Fresh() as llm, tempfile.TemporaryDirectory() as d:
        path = _write(d, _FULL)
        assert llm.load_config_file(path) == path
        assert llm.ollama_host == "http://ollama.example:11434"
        assert llm.lmstudio_base_url == "http://lmstudio.example:1234/v1"
        assert llm.openai_base_url == "https://llm.example/v1"
        assert llm.openai_api_key == "sk-from-the-file"
        assert (llm.argo_url, llm.argo_username) == ("https://argo.example/streamchat/", "someuser")
        assert llm.argo_verify == "/path/to/ca.pem"


def test_priority_flag_over_file_over_environment_over_default():
    env = {"OPENAI_BASE_URL": "https://from-env/v1", "LASSI_ARGO_USERNAME": "envuser", "OLLAMA_HOST": "http://env-host:1"}
    with _Fresh() as llm:
        assert llm.openai_base_url == "https://api.openai.com/v1"                         # default
    with _Fresh(**env) as llm:
        assert llm.openai_base_url == "https://from-env/v1"                              # environment beats default
        with tempfile.TemporaryDirectory() as d:
            llm.load_config_file(_write(d, 'openai_compatible:\n  base_url: "https://from-file/v1"\n'))
        assert llm.openai_base_url == "https://from-file/v1"                             # file beats environment
        assert llm.argo_username == "envuser" and llm.ollama_host == "http://env-host:1"  # unset in file: env stays
        llm.configure(openai_base_url="https://from-flag/v1")
        assert llm.openai_base_url == "https://from-flag/v1"                             # configure()/flag beats file


def test_blank_values_never_wipe_other_sources():
    with _Fresh(LASSI_ARGO_URL="https://env-argo/") as llm, tempfile.TemporaryDirectory() as d:
        llm.load_config_file(_write(d, 'argo:\n  url: ""\n  username: "u"\n'))
        assert llm.argo_url == "https://env-argo/" and llm.argo_username == "u"


def test_example_file_is_valid_and_changes_nothing():
    """Copying config.yaml.example unchanged must not alter any setting (all values blank or default)."""
    with _Fresh() as llm:
        before = _settings()
        assert llm.load_config_file(os.path.join(REPO, "config.yaml.example"))
        assert _settings() == before


def test_api_key_in_file_reaches_the_request():
    """The OpenAI-compatible backend sends the key from config.yaml as a bearer token."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(self.headers.get("Authorization"))
            body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with _Fresh(OPENAI_API_KEY="sk-from-env") as llm, tempfile.TemporaryDirectory() as d:
            llm.load_config_file(_write(d, f'openai_compatible:\n  base_url: "http://127.0.0.1:{server.server_port}"\n'
                                           '  api_key: "sk-from-the-file"\n'))
            llm.call_llm("openai_compatible", "m", 0, "s", "p")
            assert seen == ["Bearer sk-from-the-file"], seen         # the file wins over $OPENAI_API_KEY
            llm.configure(openai_api_key="")
            llm.call_llm("openai_compatible", "m", 0, "s", "p")
            assert seen[-1] == "Bearer sk-from-env", seen            # blank everywhere else: environment fallback
    finally:
        server.shutdown()


def test_discovery_order():
    with _Fresh() as llm, tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        os.chdir(d)
        try:
            os.environ.pop("LASSI_CONFIG")
            assert llm.find_config_file() in (None, os.path.join(REPO, "config.yaml"))   # none in this folder
            local = _write(d, "argo:\n  username: u\n")
            assert os.path.samefile(llm.find_config_file(), local)                       # ./config.yaml is found
            os.environ["LASSI_CONFIG"] = ""
            assert llm.find_config_file() is None                                        # empty = use none
            other = _write(d, "argo:\n  username: v\n", name="other.yaml")
            os.environ["LASSI_CONFIG"] = other
            assert llm.find_config_file() == other                                       # $LASSI_CONFIG
            assert llm.find_config_file(local) == local                                  # explicit path wins
        finally:
            os.chdir(old)


def test_mistakes_get_clear_errors():
    with _Fresh() as llm, tempfile.TemporaryDirectory() as d:
        cases = [
            ("argo:\n  usernmae: x\n", "unknown setting argo.usernmae"),
            ("argoo:\n  url: x\n", "unknown section 'argoo'"),
            ("argo:\n  url: 123\n", "must be text"),
            ("- just\n- a list\n", "expected sections"),
            ("openai_compatible: hello\n", "must contain"),
        ]
        for text, expected in cases:
            try:
                llm.load_config_file(_write(d, text))
                raise AssertionError(f"expected an error for: {text!r}")
            except ValueError as e:
                assert expected in str(e), (expected, str(e))
        try:
            llm.load_config_file(os.path.join(d, "missing.yaml"))
            raise AssertionError("a missing explicit file must be an error")
        except FileNotFoundError:
            pass
        assert llm.load_config_file(_write(d, "")) is not None                           # an empty file is fine


def test_cli_flags_override_the_file_and_config_option():
    from lassi.cli import build_parser, config_from_args
    with _Fresh() as llm, tempfile.TemporaryDirectory() as d:
        path = _write(d, _FULL)
        args = build_parser().parse_args(["--config", path, "--argo-username", "flaguser"])
        config_from_args(args)
        assert llm.argo_username == "flaguser"                    # flag beats file
        assert llm.argo_url == "https://argo.example/streamchat/"  # file value kept where no flag was given
        assert llm.openai_api_key == "sk-from-the-file"
        # a flag-less run leaves the file's values alone (flags default to "not given", not to blank)
        config_from_args(build_parser().parse_args(["--config", path]))
        assert llm.argo_username == "someuser"


def test_config_yaml_is_not_tracked_by_git():
    out = subprocess.run(["git", "check-ignore", "config.yaml"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, "config.yaml must be listed in .gitignore"
    out = subprocess.run(["git", "check-ignore", "config.yaml.example"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 1, "config.yaml.example must stay tracked"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS  {name}")
