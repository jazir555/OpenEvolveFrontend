import streamlit as st
import pandas as pd
import time
import requests
from openevolve_integration import OpenEvolveAPI


def render_analytics_dashboard():
    st.title("📊 Evolution Analytics Dashboard")

    if "evolution_id" not in st.session_state or not st.session_state.evolution_id:
        st.info(
            "No active evolution run to display analytics for. Start an evolution run first."
        )
        return

    evolution_id = st.session_state.evolution_id
    st.subheader(f"Evolution ID: `{evolution_id}`")

    api = OpenEvolveAPI(
        base_url=st.session_state.openevolve_base_url,
        api_key=st.session_state.openevolve_api_key,
    )

    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh (every 5 seconds)", value=False)

    # Refresh button
    if st.button("Refresh Data") or auto_refresh:
        # Fetch and display status
        status = api.get_evolution_status(evolution_id)
        if status:
            st.subheader("Current Status")
            st.json(status)
        else:
            st.warning("Could not fetch evolution status.")

        # Fetch and display best solution
        best_solution = api.get_best_solution(evolution_id)
        if best_solution:
            st.subheader("Best Solution Found")
            st.json(best_solution)
        else:
            st.info("No best solution available yet.")

        # Fetch and display history
        history = api.get_evolution_history(evolution_id)
        if history:
            st.subheader("Evolution History")
            df = pd.DataFrame(history)
            if not df.empty:
                st.line_chart(
                    df.set_index("iteration")[["best_score", "average_score"]]
                )
                # Display diversity metrics if available
                if "diversity_score" in df.columns:
                    st.line_chart(df.set_index("iteration")[["diversity_score"]])
            else:
                st.info("No history data available yet.")
        else:
            st.warning("Could not fetch evolution history.")

        # Fetch and display logs (snapshot)
        logs = []
        for chunk in api.stream_evolution_logs(evolution_id):
            logs.append(chunk)
        if logs:
            st.subheader("Evolution Logs")
            st.text_area("Logs", value="".join(logs), height=300)
        else:
            st.info("No logs available yet.")

        # Fetch and display artifacts
        artifacts = api.get_artifacts(evolution_id)
        if artifacts:
            st.subheader("Evolution Artifacts")
            with st.expander("View Artifacts"):
                for artifact in artifacts:
                    st.write(f"**Name:** {artifact.get('name')}")
                    st.write(f"**Type:** {artifact.get('type')}")
                    st.write(f"**URL:** {artifact.get('url')}")
                    st.download_button(
                        label=f"Download {artifact.get('name')}",
                        data=requests.get(artifact.get("url")).content,
                        file_name=artifact.get("name"),
                        mime=artifact.get("type"),
                    )
                    st.markdown("---")
        else:
            st.info("No artifacts available yet.")

        if auto_refresh:
            time.sleep(5)
            st.rerun()
