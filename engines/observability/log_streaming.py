from __future__ import annotations

from ui_shim import ui as st

# Optional imports with fallbacks
try:
    from flask import Flask, Response, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    Flask = None
    Response = None
    jsonify = None
    FLASK_AVAILABLE = False

import queue
import threading
from datetime import datetime
import json
import logging
from typing import Dict, Any
import pandas as pd


class LogStreaming:
    def __init__(self):
        if not FLASK_AVAILABLE:
            print("Flask not available. Log streaming features will be disabled.")
            self.app = None
            return
            
        self.log_queue = queue.Queue()
        self.app = Flask(__name__)
        self._setup_routes()
        self.flask_thread = None
        self.log_history = []  # Store recent logs
        self.max_history = 1000  # Limit log history to prevent memory issues
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('openevolve.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        if not FLASK_AVAILABLE:
            return
            
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

        @self.app.route("/health")
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

        @self.app.route("/logs/history")
        def get_log_history():
            return jsonify({"logs": self.log_history})

        @self.app.route("/metrics")
        def get_metrics():
            # Return current metrics
            metrics = {
                "log_queue_size": self.log_queue.qsize(),
                "log_history_count": len(self.log_history),
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(metrics)

    def run_flask_app_in_thread(self):
        if not FLASK_AVAILABLE:
            print("Flask not available. Cannot start log streaming service.")
            return
            
        if not FLASK_AVAILABLE:
            print("Flask not available. Cannot start log streaming service.")
            return
            
        if self.flask_thread is None or not self.flask_thread.is_alive():
            self.flask_thread = threading.Thread(
                target=self.app.run, 
                kwargs={'port': 5001, 'use_reloader': False, 'debug': False}, 
                daemon=True
            )
            self.flask_thread.start()
            st.session_state.log_streaming_flask_running = True
            self.logger.info("Log streaming service started on port 5001")

    def add_log_message(self, message: str, level: str = "INFO"):
        """Add a log message to the queue and history."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        # Add to queue for streaming
        self.log_queue.put(formatted_message)
        
        # Add to history (maintain max size)
        self.log_history.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })
        
        if len(self.log_history) > self.max_history:
            self.log_history = self.log_history[-self.max_history:]
        
        # Also log to file
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def get_logs_as_dataframe(self) -> pd.DataFrame:
        """Get log history as a pandas DataFrame for display."""
        return pd.DataFrame(self.log_history)

    def display_log_dashboard(self):
        """Display the log dashboard in UI."""
        st.subheader("Log Streaming Dashboard")
        
        if not FLASK_AVAILABLE:
            st.warning("Flask not available. Install with: pip install flask")
            return
        
        # Start/Stop controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Log Streaming Service"):
                self.run_flask_app_in_thread()
                st.success("Log streaming service started on port 5001")
        
        with col2:
            if st.button("Stop Log Streaming Service"):
                st.info("Service will stop when app closes. Restart UI for clean state.")
        
        # Display recent logs
        st.subheader("Recent Logs")
        if self.log_history:
            df = self.get_logs_as_dataframe()
            st.dataframe(df.tail(100), use_container_width=True)
        else:
            st.info("No logs yet. Start the service to begin logging.")
        
        # Display metrics
        st.subheader("Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Queue Size", self.log_queue.qsize())
        with metrics_col2:
            st.metric("History Count", len(self.log_history))

