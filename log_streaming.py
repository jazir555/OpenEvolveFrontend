import streamlit as st

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
                        yield f"data: {message}\\n\\n"
                    except queue.Empty:
                        # Send a comment to keep the connection alive
                        yield ": keep-alive\\n\\n"
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
        elif level == "INFO":
            self.logger.info(message)
        else:
            self.logger.debug(message)

    def log_evolution_event(self, event_type: str, details: Dict[str, Any]):
        """Log evolution-related events with structured data."""
        message = f"EVOLUTION {event_type.upper()}: {json.dumps(details)}"
        self.add_log_message(message, "INFO")
        
        # Also log specific metrics if available
        if "fitness" in details:
            self.add_log_message(f"Fitness score: {details['fitness']}", "INFO")
        if "generation" in details:
            self.add_log_message(f"Generation: {details['generation']}", "INFO")

    def log_adversarial_event(self, event_type: str, details: Dict[str, Any]):
        """Log adversarial testing events with structured data."""
        message = f"ADVERSARIAL {event_type.upper()}: {json.dumps(details)}"
        self.add_log_message(message, "INFO")
        
        # Log specific metrics
        if "approval_rate" in details:
            self.add_log_message(f"Approval rate: {details['approval_rate']:.2%}", "INFO")

    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system events with structured data."""
        message = f"SYSTEM {event_type.upper()}: {json.dumps(details)}"
        self.add_log_message(message, "INFO")

    def render_log_streaming_ui(self):
        st.header("📄 Log Streaming & Monitoring")
        
        if not FLASK_AVAILABLE:
            st.warning("Flask not available. Log streaming features are disabled. Install Flask to enable real-time log streaming.")
            # Basic log functionality without Flask
            st.subheader("Recent Logs")
            if self.log_history:
                # Create a dataframe for display
                recent_logs = self.log_history[-20:]  # Show last 20 logs
                log_df = []
                for log in recent_logs:
                    log_df.append({
                        "Timestamp": log["timestamp"],
                        "Level": log["level"],
                        "Message": log["message"]
                    })
                
                if log_df:
                    df = pd.DataFrame(log_df)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent logs to display.")
            return
        
        # Interactive log streaming controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Restart Log Service"):
                if self.flask_thread and self.flask_thread.is_alive():
                    st.info("Service restarted! (Note: In a real app, we'd properly restart the service)")
                else:
                    self.run_flask_app_in_thread()
                    st.success("Log streaming service restarted!")

        with col2:
            log_message = st.text_input("Custom log message", placeholder="Enter message to send to log stream...")
            log_level = st.selectbox("Log Level", ["INFO", "WARNING", "ERROR", "SUCCESS"], index=0)
            if st.button("Send Log Message"):
                if log_message.strip():
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    formatted_message = f"[{timestamp}] {log_message}"
                    self.add_log_message(formatted_message, log_level)
                    st.success(f"Message sent: {formatted_message}")
                else:
                    st.warning("Please enter a log message to send")
        
        with col3:
            # Show log metrics
            st.metric("Queue Size", self.log_queue.qsize())
            st.metric("History Count", len(self.log_history))
        
        # Show the log stream using an iframe
        st.subheader("Live Log Stream")
        st.markdown('<iframe src="http://localhost:5001/logs" width="100%" height="300px"></iframe>', unsafe_allow_html=True)
        
        st.info("💡 Tip: This shows real-time logs from the backend. The log stream updates automatically as new messages are added.")
        
        # Show recent logs
        st.subheader("Recent Logs")
        if self.log_history:
            # Create a dataframe for display
            recent_logs = self.log_history[-20:]  # Show last 20 logs
            log_df = []
            for log in recent_logs:
                log_df.append({
                    "Timestamp": log["timestamp"],
                    "Level": log["level"],
                    "Message": log["message"]
                })
            
            if log_df:
                df = pd.DataFrame(log_df)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent logs to display.")
        
        # Add sample log messages with different types
        st.subheader("Add Sample Log Messages")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            if st.button("✅ Success Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message("Operation completed successfully", "SUCCESS")
                st.success("Success log added")
        
        with col4:
            if st.button("⚠️ Warning Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message("Potential issue detected", "WARNING")
                st.success("Warning log added")
        
        with col5:
            if st.button("❌ Error Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message("Operation failed", "ERROR")
                st.success("Error log added")
        
        with col6:
            if st.button("🔄 Info Log"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.add_log_message("System update completed", "INFO")
                st.success("Info log added")
        
        # Status information
        st.subheader("Service Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.write(f"✅ Service Running: {st.session_state.get('log_streaming_flask_running', False)}")
        with status_col2:
            st.write(f"📊 Queue Size: {self.log_queue.qsize()}")
        with status_col3:
            st.write(f"📖 History: {len(self.log_history)} entries")
        
        st.info("The log streaming service allows real-time monitoring of backend operations. Perfect for debugging and monitoring!")
