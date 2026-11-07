"""
Argo Gateway smoke test: one tiny request through the real `argo` backend.

Skipped unless you configure it (Argonne network access is required). Nothing here is hardcoded:

    export LASSI_ARGO_URL=<the gateway's streamchat URL>
    export LASSI_ARGO_USERNAME=<your Argonne username>       # not a secret
    export LASSI_ARGO_MODEL=gpt4o                            # optional, default gpt4o
    python tests/test_argo_smoke.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_runner import REPO, _stub_missing_modules


def test_argo_round_trip():
    url, user = os.environ.get("LASSI_ARGO_URL"), os.environ.get("LASSI_ARGO_USERNAME")
    if not (url and user):
        raise unittest.SkipTest("set LASSI_ARGO_URL and LASSI_ARGO_USERNAME to run this test")
    _stub_missing_modules()
    sys.path.insert(0, REPO)
    from lassi import llm_backends

    llm_backends.configure(argo_url=url, argo_username=user)
    reply = llm_backends.call_llm("argo", os.environ.get("LASSI_ARGO_MODEL", "gpt4o"), 0,
                                  "You are terse.", "Reply with the single word PONG.")
    assert "PONG" in reply.upper(), reply


if __name__ == "__main__":
    try:
        test_argo_round_trip()
        print("PASS  test_argo_round_trip")
    except unittest.SkipTest as e:
        print(f"SKIP  test_argo_round_trip: {e}")
