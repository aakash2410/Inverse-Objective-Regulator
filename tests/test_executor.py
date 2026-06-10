import httpx

from ior.execution.executor import Executor, ProbeResult
from ior.targeting.probe_targeter import Probe


def test_executor_parses_openai_response(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "agent reply"}}]}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    probe = Probe(prompt="hello", divergence_dimension="x", divergence_magnitude=0.3)
    ex = Executor(endpoint="https://api.example.com/v1", api_key="k", model="m")
    result = ex.run(probe)

    assert isinstance(result, ProbeResult)
    assert result.response == "agent reply"
    assert result.violation_detected is False
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["json"]["messages"][0]["content"] == "hello"
    assert captured["headers"]["Authorization"] == "Bearer k"
