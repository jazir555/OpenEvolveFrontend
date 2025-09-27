import streamlit as st
from openevolve_integration import OpenEvolveAPI

def render_evaluator_uploader():
    st.header("Custom Evaluators")

    st.write("Upload your own evaluator functions to use in your evolutionary runs.")

    evaluator_code = st.text_area("Evaluator Code", height=300)

    if st.button("Upload Evaluator"):
        if evaluator_code:
            api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
            evaluator_id = api.upload_evaluator(evaluator_code)
            if evaluator_id:
                st.success(f"Evaluator uploaded successfully! Evaluator ID: {evaluator_id}")
            else:
                st.error("Failed to upload evaluator.")
        else:
            st.warning("Please paste your evaluator code.")

    st.header("Available Custom Evaluators")

    api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
    evaluators = api.get_custom_evaluators()

    if evaluators:
        for evaluator_id, evaluator_code in evaluators.items():
            with st.expander(f"Evaluator ID: {evaluator_id}"):
                st.code(evaluator_code, language="python")
    else:
        st.info("No custom evaluators found.")