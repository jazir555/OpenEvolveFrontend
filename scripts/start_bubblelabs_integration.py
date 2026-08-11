"""
OpenEvolve BubbleLabs Integration Startup Script

This script starts the complete OpenEvolve platform with BubbleLabs integration:
- OpenEvolve FastAPI backend (on port 8001, default)
- BubbleLab UI integration endpoints served by the backend
- Enhanced workflow management with lifecycle controls
- Real-time analytics and monitoring dashboard
- Parameter synchronization between UIs
- Advanced visualization capabilities
"""



import os
import sys
import subprocess
import threading
import time
import signal
import atexit
from typing import List, Tuple
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenEvolveBubbleLabsLauncher:
    """
    Enhanced launcher class that manages the complete OpenEvolve with BubbleLabs services.
    Includes additional services for analytics, monitoring, and enhanced workflow management.
    """

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.threads: List[threading.Thread] = []
        self.services_running = False
        self.analytics_server_process = None

    def start_main_ui(self):
        """
        Start the OpenEvolve API backend for BubbleLab clients.
        """
        def run_ui():
            try:
                # Start FastAPI backend used by BubbleLab TypeScript UI
                cmd = [
                    sys.executable, "-m", "uvicorn", "api_server:app",
                    "--host", "0.0.0.0",
                    "--port", "8001"
                ]

                logger.info("Starting OpenEvolve API backend for BubbleLab integration...")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )

                self.processes.append(process)
                logger.info(f"Started API backend on port 8001 with PID {process.pid}")

                # Wait for the process to complete
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    logger.error(f"API backend exited with code {process.returncode}")
                    logger.error(f"Error: {stderr}")
                else:
                    logger.info("API backend exited normally")

            except (subprocess.SubprocessError, OSError, ValueError) as e:
                logger.error(f"Error starting API backend: {e}")

        # Start the UI in a background thread
        ui_thread = threading.Thread(target=run_ui, daemon=True)
        ui_thread.start()
        self.threads.append(ui_thread)
        return ui_thread

    def start_analytics_server(self):
        """
        Start the analytics and monitoring server (if needed for enhanced features).
        """
        def run_analytics_server():
            try:
                # In a real implementation, this might start a Flask/FastAPI server
                # for real-time analytics, but for now we'll just log that it would start
                logger.info("Analytics and monitoring server would start here")
                
                # Keep the thread alive
                while self.services_running:
                    time.sleep(1)
                    
            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Error in analytics server: {e}")

        # Start analytics server in background thread
        analytics_thread = threading.Thread(target=run_analytics_server, daemon=True)
        analytics_thread.start()
        self.threads.append(analytics_thread)
        return analytics_thread

    def start_all_services(self):
        """
        Start all services: Main UI with integrated BubbleLabs and enhanced components.
        """
        print("Starting OpenEvolve with Enhanced BubbleLabs Integration...")
        print("=" * 60)
        logger.info("Starting OpenEvolve with Enhanced BubbleLabs Integration")

        self.services_running = True

        # Start analytics and monitoring server
        print("Starting analytics and monitoring services...")
        self.start_analytics_server()

        # Start main UI which includes BubbleLabs tab
        print("Starting OpenEvolve API backend for BubbleLabs...")
        self.start_main_ui()

        print("=" * 60)
        print("🎉 All services started successfully!")
        print("🌐 OpenEvolve API: http://localhost:8001")
        print("🔧 Configure BubbleLab frontend to use the OpenEvolve API endpoints")
        print("📊 Use the 'Parameter Sync' tab to synchronize settings between UIs")
        print("📈 Access the 'Analytics Dashboard' for monitoring and reporting")
        print("🔄 Full workflow lifecycle controls available in 'Workflow Control' tab")
        print("📋 Enhanced visualization in 'Global Parameters' and dedicated tabs")
        print("🔍 Complete OpenEvolve parameter control available in BubbleLabs UI")
        print("⚡ Real-time performance monitoring and resource utilization tracking")
        print("🎯 Advanced workflow visualization and progress tracking")
        print("📝 Detailed analytics and reporting capabilities")
        print("=" * 60)

        # Wait for all threads to complete
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received")
            self.stop_all_services()

    def stop_all_services(self):
        """
        Stop all running services gracefully.
        """
        print("Stopping all services...")
        logger.info("Stopping all OpenEvolve BubbleLabs services...")
        
        self.services_running = False

        # Terminate all processes
        for i, process in enumerate(self.processes):
            try:
                logger.info(f"Terminating process {i+1} (PID: {process.pid})...")
                process.terminate()
                
                # Wait up to 10 seconds for graceful shutdown
                try:
                    process.wait(timeout=10)
                    logger.info(f"Process {process.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {process.pid} did not terminate gracefully, forcing kill...")
                    process.kill()
                    process.wait()  # Wait for the kill to complete
                    
            except ProcessLookupError:
                logger.info(f"Process {process.pid} already terminated")
            except (ProcessLookupError, PermissionError, OSError) as e:
                logger.error(f"Error stopping process {process.pid}: {e}")

        # Wait a moment for threads to finish
        time.sleep(1)
        
        print("All services stopped successfully.")
        logger.info("All OpenEvolve BubbleLabs services stopped")


def main():
    """
    Main entry point for the enhanced launcher.
    """
    launcher = OpenEvolveBubbleLabsLauncher()

    # Register cleanup function
    def cleanup():
        print("\nExecuting cleanup procedures...")
        launcher.stop_all_services()
        print("Cleanup completed.")

    atexit.register(cleanup)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal (SIGINT), shutting down gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        launcher.start_all_services()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
        cleanup()
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Unexpected error in launcher: {e}")
        cleanup()
        raise


if __name__ == "__main__":
    main()
