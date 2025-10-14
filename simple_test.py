import os
import sys
import subprocess
import time
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_project_root():
    """
    Returns the absolute path to the project's root directory.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return current_dir
    except Exception as e:
        logging.error(f"Error getting project root: {e}")
        return os.getcwd()

def start_openevolve_backend():
    """
    Starts the OpenEvolve backend.
    """
    try:
        backend_path = os.path.join(get_project_root(), "openevolve")
        backend_script = os.path.join(backend_path, "scripts", "visualizer.py")

        if not os.path.exists(backend_script):
            logging.error(f"Backend script not found at {backend_script}")
            return

        command = [sys.executable, backend_script, "--port", "8080"]
        env = os.environ.copy()
        env["PYTHONPATH"] = get_project_root()

        process = subprocess.Popen(
            command,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )

        logging.info(f"OpenEvolve backend started with PID: {process.pid}")

        time.sleep(5)  # Wait for the server to start

        # Health check
        response = requests.get("http://localhost:8080/")
        if response.status_code == 200:
            logging.info("OpenEvolve backend is running.")
        else:
            logging.error(f"Backend health check failed with status code: {response.status_code}")

    except Exception as e:
        logging.error(f"Failed to start OpenEvolve backend: {e}")

if __name__ == "__main__":
    start_openevolve_backend()
