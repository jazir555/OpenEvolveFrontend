"""Multimodal determinism helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MultiModalGenerator:
    text_generator: Optional[Callable[..., Any]] = None
    vision_encoder: Optional[Callable[[str], str]] = None
    llm_generate: Optional[Callable[..., Any]] = None

    def generate_from_image(self, image_path: str, text_prompt: str) -> Any:
        encoder = self.vision_encoder or (lambda path: f"[image:{path}]")
        embedding = encoder(image_path)
        prompt = f"Image: {embedding}\n{text_prompt}"
        if self.llm_generate:
            return self.llm_generate(prompt=prompt)
        if self.text_generator:
            return self.text_generator(prompt=prompt)
        return {"prompt": prompt}


class MultiModalDeterministicGenerator:
    def __init__(self, text_generator: Callable[..., Any], image_generator: Callable[..., Any], verifier: Optional[Callable[..., bool]] = None):
        self.text_generator = text_generator
        self.image_generator = image_generator
        self.verifier = verifier or (lambda _: True)

    def generate_multimodal(self, prompt: str, modalities: List[str], consistency_constraints: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for modality in modalities:
            if modality == "text":
                results["text"] = self.text_generator(prompt=prompt, schema=consistency_constraints.get("text_schema"))
            elif modality == "image":
                results["image"] = self.image_generator(prompt=prompt, constraints=consistency_constraints.get("image_constraints"))
            else:
                raise ValueError(f"Unsupported modality: {modality}")
        results["verified"] = self.verifier(results)
        return results
