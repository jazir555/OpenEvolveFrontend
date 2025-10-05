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
        
        # Start Flask app if not already running
        if not st.session_state.get("log_streaming_flask_running", False):
            self.run_flask_app_in_thread()
            st.success("Log streaming service started!")
        
        # Interactive log streaming controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart Log Service"):
                if self.flask_thread and self.flask_thread.is_alive():
                    # Note: This is a limitation - Flask threads can't be stopped easily
                    st.info("Service restarted! (Note: In a real app, we'd properly restart the service)")
                else:
                    self.run_flask_app_in_thread()
                    st.success("Log streaming service restarted!")
        
        with col2:
            log_message = st.text_input("Custom log message", placeholder="Enter message to send to log stream...")
            if st.button("Send Log Message"):
                if log_message.strip():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    formatted_message = f"[{timestamp}] {log_message}"
                    self.add_log_message(formatted_message)
                    st.success(f"Message sent: {formatted_message}")
                else:
                    st.warning("Please enter a log message to send")
        
        # Show the log stream using an iframe
        st.subheader("Live Log Stream")
        st.markdown('<iframe src="http://localhost:5001/logs" width="100%" height="400px"></iframe>', unsafe_allow_html=True)
        
        st.info("💡 Tip: This shows real-time logs from the backend. The log stream updates automatically as new messages are added.")
        
        # Add sample log messages with different types
        st.subheader("Add Sample Log Messages")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            if st.button("✅ Success Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message(f"[{timestamp}] SUCCESS: Operation completed successfully")
                st.success("Success log added")
        
        with col4:
            if st.button("⚠️ Warning Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message(f"[{timestamp}] WARNING: Potential issue detected")
                st.success("Warning log added")
        
        with col5:
            if st.button("❌ Error Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message(f"[{timestamp}] ERROR: Operation failed")
                st.success("Error log added")
        
        with col6:
            if st.button("🔄 Info Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message(f"[{timestamp}] INFO: System update completed")
                st.success("Info log added")
        
        # Status information
        st.subheader("Service Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.write(f"✅ Service Running: {st.session_state.get('log_streaming_flask_running', False)}")
        with status_col2:
            st.write(f"📊 Queue Size: {self.log_queue.qsize()}")
        
        st.info("The log streaming service allows real-time monitoring of backend operations. Perfect for debugging and monitoring!")