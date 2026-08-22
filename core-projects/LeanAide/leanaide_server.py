import multiprocessing
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from server.logging_utils import (create_env_file, delete_env_file,
                                  filter_logs, log_write)

LEANAIDE_PORT = int(os.environ.get("LEANAIDE_PORT", 7654))

home_dir = str(Path(__file__).resolve().parent)
serv_dir = os.path.join(home_dir, "server")

SERVER_FILE = os.path.join(serv_dir, "api_server.py")
COMMAND = os.environ.get("LEANAIDE_COMMAND", "lake exe leanaide_process")

for arg in sys.argv[1:]:
    if arg not in ["--ui", "--no-server", "--ns", "--clear-cache", "--help", "-h"]:
        COMMAND += " " + arg

# Ensure the environment file exists
create_env_file(fresh=True)

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_server_api():
    """Run the backend REST API server using uvicorn"""
    process = subprocess.Popen(
        [sys.executable, SERVER_FILE, COMMAND],
        stderr=subprocess.PIPE,
        text=True
    )

    # Log stderr
    if process.stdout:
        for line in process.stdout:
            line = filter_logs(line.strip())
            print(line)
            log_write("Server Stdout", line, True)

    if process.stderr:
        for line in process.stderr:
            line = filter_logs(line.strip())
            print(line, file=sys.stderr)
            log_write("Server Stderr", line, True)

def signal_handler(sig, frame):
    """Handle Ctrl+C to terminate the server"""
    print("\nShutting down server...", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            help_text = """
Usage: leanaide_server.py [FLAGS] [LEANAIDE_PROCESS_FLAGS]

Server FLAGS:
    --help | -h            Show this help message
    --ui                   Ignored. The web UI is provided by BubbleLab
                           (TypeScript) at core-projects/BubbleLab.
    --no-server | --ns     Don't run the backend server
    --clear-cache          Clear the cache before starting the server

The web UI for LeanAide is provided by BubbleLab (TypeScript), located at
core-projects/BubbleLab. This launcher starts only the backend REST API
server, which listens on port 7654.

LEANAIDE_PROCESS_FLAGS (passed to `lake exe leanaide_process`):
    -h, --help                    Prints this message.
    --include_fixed               Include the 'Lean Chat' fixed prompts.
    -p, --prompts : Nat           Number of example prompts (default 20).
    --descriptions : Nat          Number of example descriptions (default 2).
    --concise_descriptions : Nat  Number of example concise descriptions
                    (default 2).
    --leansearch_prompts : Nat    Number of examples from LeanSearch
    --moogle_prompts : Nat        Number of examples from Moogle
    -n, --num_responses : Nat     Number of responses to ask for (default 10).
    -t, --temperature : Nat       Scaled temperature `t*10` for temperature `t`
                    (default 8).
    -m, --model : String          Model to be used (default `gpt-5.1`)
    --azure                       Use Azure instead of OpenAI.
    --gemini                      Use Gemini with OpenAI API.
    --url : String                URL to query (for a local server).
    --examples_url : String       URL to query for nearby embeddings (for a
                    generic server).
    --auth_key : String           Authentication key (for a local or generic
                    server).
    --show_prompt                 Output the prompt to the LLM.
    --show_elaborated             Output the elaborated terms
    --max_tokens : Nat            Maximum tokens to use in the translation.
    --no_sysprompt                The model has no system prompt (not relevant
                    for GPT models).
"""
            print(help_text)
            sys.exit(0)

    run_server = "--no-server" not in sys.argv and "--ns" not in sys.argv

    if "--clear-cache" in sys.argv:
        cache_dir = os.path.join(home_dir, ".leanaide_cache")
        try:
            for subdir in os.listdir(cache_dir):
                for item in subdir:
                    item_path = os.path.join(cache_dir, subdir, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)

            print("Cache cleared successfully. Starting fresh.", file=sys.stderr)
            for dirname in ["prompt", "chat"]:
                os.makedirs(os.path.join(cache_dir, dirname), exist_ok=True)
        except Exception as e:
            print(f"Error clearing cache: {e}", file=sys.stderr)

    print("\n\033[1;34mStarting server:\033[0m", file=sys.stderr)

    # Start the backend API server
    serv_process = None

    if run_server:
        host_display = os.environ.get('HOST', '0.0.0.0')
        if host_display == '0.0.0.0':
            host_display = socket.gethostname()  # Show actual hostname instead of 0.0.0.0
        print(f"\033[1;34mAPI Server:\033[0m http://{host_display}:{LEANAIDE_PORT}\n", file=sys.stderr)
        serv_process = multiprocessing.Process(target=run_server_api)
        serv_process.start()
    else:
        print("\033[1;34mRunning without API server\033[0m\n", file=sys.stderr)

    try:
        # Keep main process alive
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")
        if serv_process:
            serv_process.terminate()
            serv_process.join()

    delete_env_file()  # Clean up environment file
