from flask import Flask, Response
import queue

log_queue = queue.Queue()

app = Flask(__name__)


@app.route("/logs")
def stream_logs():
    def generate():
        while True:
            try:
                message = log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Send a comment to keep the connection alive
                yield ": keep-alive\n\n"

    return Response(generate(), mimetype="text/event-stream")


def run_flask_app():
    app.run(port=5001)
