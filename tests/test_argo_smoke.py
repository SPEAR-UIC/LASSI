"""
Argo Gateway smoke test: one tiny request through the real `argo` backend.

Skipped unless Argo is configured (Argonne network access is required). Configure it the way you
normally would, in config.yaml:

    argo:
      url: "<the gateway's streamchat URL>"
      username: "<your Argonne username>"      # not a secret

or with the environment variables LASSI_ARGO_URL and LASSI_ARGO_USERNAME. Optionally set
LASSI_ARGO_MODEL (default gpt4o). Nothing is hardcoded in the repository.

    python tests/test_argo_smoke.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, USER_LASSI_CONFIG, _stub_missing_modules

# The shared test runner isolates tests from config.yaml; this test is the one that needs it.
if USER_LASSI_CONFIG is None:
    os.environ.pop("LASSI_CONFIG", None)
else:
    os.environ["LASSI_CONFIG"] = USER_LASSI_CONFIG


def test_argo_round_trip():
    _stub_missing_modules()
    sys.path.insert(0, REPO)
    from lassi import llm_backends

    if not (llm_backends.argo_url and llm_backends.argo_username):
        raise unittest.SkipTest("Argo is not configured: set argo.url and argo.username in config.yaml "
                                "(or LASSI_ARGO_URL and LASSI_ARGO_USERNAME)")
    reply = llm_backends.call_llm("argo", os.environ.get("LASSI_ARGO_MODEL", "gpt4o"), 0,
                                  "You are terse.", "Reply with the single word PONG.")
    assert "PONG" in reply.upper(), reply


if __name__ == "__main__":
    try:
        test_argo_round_trip()
        print("PASS  test_argo_round_trip")
    except unittest.SkipTest as e:
        print(f"SKIP  test_argo_round_trip: {e}")
