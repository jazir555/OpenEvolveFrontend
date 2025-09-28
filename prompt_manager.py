import streamlit as st
from openevolve_integration import OpenEvolveAPI


def render_prompt_manager():
    st.header("Custom Prompts")

    st.write(
        "Create and save your own custom prompts to use in your evolutionary runs."
    )

    prompt_name = st.text_input("Prompt Name")
    prompt_content = st.text_area("Prompt Content", height=300)

    if st.button("Save Prompt"):
        if prompt_name and prompt_content:
            api = OpenEvolveAPI(
                base_url=st.session_state.openevolve_base_url,
                api_key=st.session_state.openevolve_api_key,
            )
            if api.save_custom_prompt(prompt_name, prompt_content):
                st.success(f"Prompt '{prompt_name}' saved successfully!")
            else:
                st.error("Failed to save prompt.")
        else:
            st.warning("Please enter a prompt name and content.")

    st.header("Available Custom Prompts")

    api = OpenEvolveAPI(
        base_url=st.session_state.openevolve_base_url,
        api_key=st.session_state.openevolve_api_key,
    )
    prompts = api.get_custom_prompts()

    if prompts:
        for prompt_name, prompt_content in prompts.items():
            with st.expander(f"Prompt Name: {prompt_name}"):
                st.code(prompt_content, language="text")
    else:
        st.info("No custom prompts found.")

def handle_prompt_input(prompt_text: str) -> str:
    """
    Placeholder function to handle a given prompt input.
    In a real scenario, this would process the prompt,
    potentially sending it to an API or using it in some logic.
    """
    st.write(f"Processing prompt: {prompt_text}")
    # Here you would add logic to actually handle the prompt,
    # e.g., send it to the OpenEvolveAPI, run a model, etc.
    return f"Prompt '{prompt_text}' handled successfully (placeholder response)."
