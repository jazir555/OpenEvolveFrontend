import streamlit as st
from openevolve_integration import OpenEvolveAPI

class PromptManager:
    def __init__(self, api: OpenEvolveAPI):
        self.api = api

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

    from openevolve_integration import run_unified_evolution

def handle_prompt_input(self, prompt_text: str) -> str:
    st.write(f"Processing prompt: {prompt_text}")
    try:
        model_configs = [{
            "name": st.session_state.model,
            "weight": 1.0,
            "api_key": st.session_state.api_key,
            "api_base": st.session_state.base_url,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "max_tokens": st.session_state.max_tokens,
        }]

        result = run_unified_evolution(
            content=prompt_text,
            content_type="prompt",
            evolution_mode="prompt_optimization",
            model_configs=model_configs,
            api_key=st.session_state.api_key,
            api_base=st.session_state.base_url,
            max_iterations=st.session_state.get("max_iterations", 10), # Use a smaller number of iterations for prompt optimization
            population_size=st.session_state.get("population_size", 50),
            system_message=st.session_state.get("system_prompt", "You are a helpful assistant."),
        )

        if result and result.get("success"):
            return f"Prompt evolution completed successfully!\nBest prompt: {result.get('best_prompt')}\nBest score: {result.get('best_score')}"
        else:
            return f"Prompt evolution failed: {result.get('message', 'Unknown error')}"
    except Exception as e:
        return f"An error occurred during prompt evolution: {e}"