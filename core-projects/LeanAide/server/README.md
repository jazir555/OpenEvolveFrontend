# LeanAide Backend Server

This is the backend server for LeanAide.

## About

The LeanAide backend provides a REST API for POSTING and GETTING queries for Autoformalization tasks. The main backend is powered by the LeanAide tools.

The web UI for LeanAide is provided by **BubbleLab (TypeScript)**, located at `core-projects/BubbleLab`. This repository no longer contains a Python frontend.

The backend API server listens on `http://localhost:7654`. The port can be changed via the `LEANAIDE_PORT` environment variable.

## Installation Instructions

1. If you have not cloned the repository, you can do so with the following command:

```bash
git clone https://github.com/siddhartha-gadgil/LeanAide.git
cd LeanAide # go inside LeanAide directory
```

2. Create an environment, either using default Python or [uv](https://docs.astral.sh/uv/)(recommended):

```bash
uv venv --python 3.13
```

3. Install the required packages:

```bash
uv pip install -r requirements.txt
source .venv/bin/activate
```

4. Run the Server:

```bash
python3 leanaide_server.py
```

This will start the backend API server on `http://localhost:7654`.
