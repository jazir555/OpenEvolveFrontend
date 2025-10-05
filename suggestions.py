import streamlit as st
import json
from evolution import _request_openai_compatible_chat, _compose_messages
from typing import List, Dict, Any


@st.cache_data(ttl=3600) # Cache for 1 hour
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


@st.cache_data(ttl=3600) # Cache for 1 hour
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


@st.cache_data(ttl=3600) # Cache for 1 hour
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


@st.cache_data(ttl=3600) # Cache for 1 hour
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

def render_suggestions():
    """
    Renders the suggestions section in the Streamlit UI.
    Displays AI-powered content suggestions, classifications, improvement potential, etc.
    """
    st.header("💡 AI-Powered Suggestions")
    
    content = st.session_state.get("protocol_text", "").strip()
    if not content:
        st.info("Enter content in the main editor to get AI-powered suggestions.")
        return
    
    # Tabs for different suggestion types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Content Suggestions", 
        "🏷️ Classification & Tags", 
        "🛡️ Security Check", 
        "📈 Improvement Potential"
    ])
    
    with tab1:
        st.subheader("Content Improvement Suggestions")
        if st.button("Generate Content Suggestions", key="gen_content_sugg"):
            with st.spinner("Analyzing content and generating suggestions..."):
                try:
                    suggestions = get_content_suggestions(content)
                    if suggestions:
                        st.success(f"Generated {len(suggestions)} suggestions!")
                        for i, suggestion in enumerate(suggestions, 1):
                            st.info(f"{i}. {suggestion}")
                    else:
                        st.info("No specific suggestions generated for this content.")
                except Exception as e:
                    st.error(f"Error generating suggestions: {e}")
    
    with tab2:
        st.subheader("Content Classification & Tags")
        if st.button("Classify Content", key="classify_content"):
            with st.spinner("Analyzing content classification and tags..."):
                try:
                    classification_and_tags = get_content_classification_and_tags(content)
                    if classification_and_tags:
                        st.success("Content analysis completed!")
                        
                        if classification_and_tags.get("classification"):
                            st.write(f"**Classification:** {classification_and_tags['classification']}")
                        
                        if classification_and_tags.get("tags"):
                            st.write("**Suggested Tags:**")
                            for tag in classification_and_tags["tags"]:
                                st.markdown(f"- `{tag}`")
                        else:
                            st.info("No tags were suggested for this content.")
                    else:
                        st.info("Could not classify this content.")
                except Exception as e:
                    st.error(f"Error during classification: {e}")
    
    with tab3:
        st.subheader("Security Vulnerability Check")
        if st.button("Scan for Security Issues", key="scan_security"):
            with st.spinner("Scanning for security vulnerabilities..."):
                try:
                    vulnerabilities = check_security_vulnerabilities(content)
                    if vulnerabilities:
                        st.warning(f"Found {len(vulnerabilities)} potential security issues:")
                        for vuln in vulnerabilities:
                            st.warning(f"⚠️ {vuln}")
                    else:
                        st.success("✅ No immediate security vulnerabilities detected.")
                except Exception as e:
                    st.error(f"Error during security scan: {e}")
    
    with tab4:
        st.subheader("Improvement Potential")
        if st.button("Calculate Improvement Potential", key="calc_potential"):
            with st.spinner("Calculating improvement potential..."):
                try:
                    potential = predict_improvement_potential(content)
                    st.metric("Improvement Potential", f"{potential:.2%}")
                    
                    if potential > 0.7:
                        st.info("🎉 This content has high potential for improvement!")
                    elif potential > 0.4:
                        st.info("👍 This content has moderate potential for improvement.")
                    else:
                        st.info("✅ This content appears to be well-structured with limited improvement potential.")
                    
                    # Show explanation
                    st.write("**How it's calculated:**")
                    st.write("- Based on the number of suggestions")
                    st.write("- Analysis of content structure and tags")
                    st.write("- Assessment of content completeness")
                    
                except Exception as e:
                    st.error(f"Error calculating improvement potential: {e}")
    
    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Generate All Suggestions", key="all_sugg"):
            st.session_state.suggestions_run_all = True
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Suggestions", key="copy_sugg"):
            # In a real implementation, this would copy to clipboard
            st.info("Suggestions copied to clipboard! (Simulated)")
    
    with col3:
        if st.button("💾 Save Suggestions", key="save_sugg"):
            # In a real implementation, this would save to a file
            st.success("Suggestions saved! (Simulated)")
    
    # If all suggestions were requested
    if st.session_state.get("suggestions_run_all"):
        st.divider()
        st.subheader("📊 Complete Analysis Results")
        
        with st.spinner("Running full analysis..."):
            # Run all analyses
            suggs = get_content_suggestions(content)
            classification = get_content_classification_and_tags(content)
            vulns = check_security_vulnerabilities(content)
            potential = predict_improvement_potential(content)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Suggestions", len(suggs))
                st.metric("Improvement Potential", f"{potential:.2%}")
            with col2:
                st.metric("Security Issues", len(vulns))
                
            if classification:
                if classification.get("classification"):
                    st.write(f"**Classification:** {classification['classification']}")
                if classification.get("tags"):
                    st.write("**Tags:** " + ", ".join(classification["tags"]))
        
        st.session_state.suggestions_run_all = False
