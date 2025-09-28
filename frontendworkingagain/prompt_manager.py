import streamlit as st
from openevolve_integration import OpenEvolveAPI

class PromptManager:
    def __init__(self):
        # Initialize OpenEvolveAPI here or pass it in if it's a dependency
        # For now, let's assume it can be initialized within the class
        self.api = OpenEvolveAPI(
            base_url=st.session_state.get("openevolve_base_url", "http://localhost:8000"), # Default URL
            api_key=st.session_state.get("openevolve_api_key", ""), # Default empty key
        )

    def render_prompt_manager_ui(self):
        st.header("Custom Prompts")
        st.write("Create and save your own custom prompts to use in your evolutionary runs.")

        prompt_name = st.text_input("Prompt Name")
        prompt_content = st.text_area("Prompt Content", height=300)

        if st.button("Save Prompt"):
            if prompt_name and prompt_content:
                if self.api.save_custom_prompt(prompt_name, prompt_content):
                    st.success(f"Prompt '{prompt_name}' saved successfully!")
                else:
                    st.error("Failed to save prompt.")
            else:
                st.warning("Please enter a prompt name and content.")

        st.header("Available Custom Prompts")
        prompts = self.api.get_custom_prompts()

        if prompts:
            for prompt_name, prompt_content in prompts.items():
                with st.expander(f"Prompt Name: {prompt_name}"):
                    st.code(prompt_content, language="text")
        else:
            st.info("No custom prompts found.")

    def handle_prompt_input(self, prompt_text: str) -> str:
        st.write(f"Processing prompt: {prompt_text}")
        return f"Prompt '{prompt_text}' handled successfully (placeholder response)."