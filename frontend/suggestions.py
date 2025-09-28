import streamlit as st
import json
from evolution import _request_openai_compatible_chat, _compose_messages
from typing import List, Dict, Any


def get_content_suggestions(content: str) -> List[str]:
    """
    Get AI-powered content suggestions for improvement recommendations.
    """
    api_key = st.session_state.api_key
    base_url = st.session_state.base_url
    model = st.session_state.model
    extra_headers = st.session_state.extra_headers
    temperature = st.session_state.temperature
    top_p = st.session_state.top_p
    frequency_penalty = st.session_state.frequency_penalty
    presence_penalty = st.session_state.presence_penalty
    max_tokens = st.session_state.max_tokens
    seed = st.session_state.seed

    system_prompt = "You are an AI assistant that provides suggestions for improving the given content. Provide a list of suggestions in a clear and concise manner."
    messages = _compose_messages(system_prompt, content)

    response = _request_openai_compatible_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        extra_headers=extra_headers,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )

    suggestions = response.split("\n")
    return [s.strip() for s in suggestions if s.strip()]


def get_content_classification_and_tags(content: str) -> Dict[str, Any]:
    """
    Get AI-powered content classification and tags.
    """
    api_key = st.session_state.api_key
    base_url = st.session_state.base_url
    model = st.session_state.model
    extra_headers = st.session_state.extra_headers
    temperature = st.session_state.temperature
    top_p = st.session_state.top_p
    frequency_penalty = st.session_state.frequency_penalty
    presence_penalty = st.session_state.presence_penalty
    max_tokens = st.session_state.max_tokens
    seed = st.session_state.seed

    system_prompt = "You are an AI assistant that classifies the given content and suggests relevant tags. Provide the classification and a list of tags in JSON format."
    messages = _compose_messages(system_prompt, content)

    response = _request_openai_compatible_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        extra_headers=extra_headers,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"classification": "", "tags": []}


def predict_improvement_potential(content: str) -> float:
    """
    Predict the improvement potential of the content.
    """
    # This is a simple heuristic for now. A more advanced implementation could use a trained model.
    suggestions = get_content_suggestions(content)
    classification_and_tags = get_content_classification_and_tags(content)

    score = 0
    score += len(suggestions) * 0.1
    score += len(classification_and_tags.get("tags", [])) * 0.05

    return min(1.0, score)


def check_security_vulnerabilities(content: str) -> List[str]:
    """
    Check for common security vulnerabilities in code.
    """
    api_key = st.session_state.api_key
    base_url = st.session_state.base_url
    model = st.session_state.model
    extra_headers = st.session_state.extra_headers
    temperature = st.session_state.temperature
    top_p = st.session_state.top_p
    frequency_penalty = st.session_state.frequency_penalty
    presence_penalty = st.session_state.presence_penalty
    max_tokens = st.session_state.max_tokens
    seed = st.session_state.seed

    system_prompt = "You are a security expert. Analyze the following code for common security vulnerabilities and provide a list of potential issues."
    messages = _compose_messages(system_prompt, content)

    response = _request_openai_compatible_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        extra_headers=extra_headers,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )

    vulnerabilities = response.split("\n")
    return [v.strip() for v in vulnerabilities if v.strip()]
