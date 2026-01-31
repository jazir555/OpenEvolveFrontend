import json

import pytest

from determinism_stack import DeterministicPipeline, HybridDeterministicSystem


class StaticLLM:
    def __init__(self, text: str, provider: str = "hf", model: str = "static"):
        self._text = text
        self.provider = provider
        self.model = model
        self.tokenizer = None

    def generate(self, prompt: str, **kwargs):
        return self._text

    def stream(self, prompt: str, **kwargs):
        yield self._text


class FlakyLLM:
    def __init__(self):
        self.provider = "openai"
        self.model = "flaky-model"
        self.tokenizer = None
        self._counter = 0

    def generate(self, prompt: str, **kwargs):
        self._counter += 1
        if self._counter % 2 == 0:
            return "{\"title\": \"A\"}"
        return "{\"different\": \"X\"}"

    def stream(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


def test_deterministic_pipeline_schema_validation():
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    llm = StaticLLM("{\"title\": \"OK\"}")
    pipeline = DeterministicPipeline(llm=llm)
    result = pipeline.generate_with_all_layers("Say OK", schema=schema)
    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.output.get("title") == "OK"
    assert result.validation["valid"] is True
    assert result.reproducibility is not None


def test_hybrid_falls_back_on_cloud_divergence():
    cloud_llm = FlakyLLM()
    local_llm = StaticLLM("{\"title\": \"LOCAL\"}", provider="hf", model="local")
    system = HybridDeterministicSystem(cloud_llm=cloud_llm, local_llm=local_llm)
    result = system.generate("Return JSON title", mode="hybrid")
    assert isinstance(result.output, dict) or isinstance(result.output, str)
    if isinstance(result.output, dict):
        assert result.output.get("title") == "LOCAL"
    else:
        assert json.loads(result.output).get("title") == "LOCAL"
