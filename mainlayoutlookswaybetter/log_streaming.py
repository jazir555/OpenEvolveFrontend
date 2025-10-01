import streamlit as st
from flask import Flask, Response
import queue
import threading
from datetime import datetime

class LogStreaming:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.app = Flask(__name__)
        self._setup_routes()
        self.flask_thread = None

    def _setup_routes(self):
        @self.app.route("/logs")
        def stream_logs():
            def generate():
                while True:
                    try:
                        message = self.log_queue.get(timeout=1)
                        yield f"data: {message}\n\n"
                    except queue.Empty:
                        # Send a comment to keep the connection alive
                        yield ": keep-alive\n\n"
            return Response(generate(), mimetype="text/event-stream")

    def run_flask_app_in_thread(self):
        if self.flask_thread is None or not self.flask_thread.is_alive():
            self.flask_thread = threading.Thread(target=self.app.run, kwargs={'port': 5001}, daemon=True)
            self.flask_thread.start()
            st.session_state.log_streaming_flask_running = True # Indicate that Flask is running

    def add_log_message(self, message: str):
        self.log_queue.put(message)

    def render_log_streaming_ui(self):
        st.header("📄 Log Streaming")
        st.info("Log streaming features are under development. Stay tuned!")

        # Start Flask app if not already running
        if not st.session_state.get("log_streaming_flask_running", False):
            self.run_flask_app_in_thread()
            st.rerun() # Rerun to update UI after starting Flask

        st.markdown('<iframe src="http://localhost:5001/logs" width="100%" height="400px"></iframe>', unsafe_allow_html=True)

        # Example of adding a log message
        if st.button("Add Test Log Message"):
            self.add_log_message(f"Test message from Streamlit at {datetime.now()}")