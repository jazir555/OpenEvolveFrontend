import ast
import json
import os
import re
import socket
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Tuple, Type

import requests

from api_server import HOST
from logging_utils import (LEANAIDE_HOMEDIR, log_buffer_clean, log_server,
                           log_write)

HOMEDIR = str(Path(__file__).resolve().parent.parent) # LeanAide root
sys.path.append(HOMEDIR)
schema_path = os.path.join(str(HOMEDIR), "resources", "PaperStructure.json")
SCHEMA_JSON = json.load(open(schema_path, "r", encoding="utf-8"))

TOKEN_JSON_FILE = f"{LEANAIDE_HOMEDIR}/.leanaide_cache/tasks/token_status.json"

# Lean Checker Tasks
TASKS = {
    "Echo": {
        "task_name": "echo",
        "input": {"data": "String"},
        "output": {"data": "String"},
        "commonly_used": False,
    },
    "Documentation for a Theorem": {
        "task_name": "theorem_doc",
        "input": {"theorem_name": "String", "theorem_statement": "String"},
        "output": {"theorem_doc": "String"},
        "commonly_used": False,
    },
    "Documentation for a Definition": {
        "task_name": "def_doc",
        "input": {"definition_name": "String", "definition_code": "String"},
        "output": {"definition_doc": "String"},
        "commonly_used": False,
    },
    "Translate Theorem": {
        "task_name": "translate_thm",
        "input": {"theorem_text": "String"},
        "output": {"theorem_code": "String"},
        "parameters": {
            "greedy": "Bool (default: true)",
            "fallback": "Bool (default: true)",
        },
        "commonly_used": False,
    },
    "Translate Definition": {
        "task_name": "translate_def",
        "input": {"definition_text": "String"},
        "output": {"definition_code": "String"},
        "parameters": {"fallback": "Bool (default: true)"},
        "commonly_used": False,
    },
    "Theorem Name": {
        "task_name": "theorem_name",
        "input": {"theorem_text": "String"},
        "output": {"theorem_name": "String"},
        "commonly_used": False,
    },
    "Prove": {
        "task_name": "prove",
        "input": {"theorem_text": "String"},
        "output": {"proof_text": "String"},
        "commonly_used": False,
    },
    "Translate Theorem Detailed": {
        "task_name": "translate_thm_detailed",
        "input": {"theorem_text": "String"},
        "output": {
            "theorem_code": "String",
            "theorem_name": "String",
            "proved": "Bool",
            "theorem_statement": "String",
            "definitions_used": "String"
        },
        "parameters": {
            "greedy": "Bool (default: true)",
            "fallback": "Bool (default: true)",
        },
        "commonly_used": True,
    },
    "Structured JSON Proof": {
        "task_name": "structured_json_proof",
        "input": {"theorem_text": "String", "proof_text": "String"},
        "output": {"document_json": "Json"},
        "commonly_used": False,
    },
    "Elaborate Lean Code": {
        "task_name": "elaborate",
        "input": {"document_code": "String", "declarations": "List Name"},
        "output": {"logs": "List String", "sorries": "List Json"},
        "parameters": {
            "top_code": 'String (default: "")',
            "describe_sorries": "Bool (default: false)",
        },
        "commonly_used": True,
    },
    "Lean from JSON Structured": {
        "task_name": "lean_from_json_structured",
        "input": {"document_json": "Json"},
        "output": {
            "document_code": "String",
            "declarations": "List String",
            "top_code": "String",
        },
        "commonly_used": True,
    },
}

def get_actual_input(input_str: str) -> Tuple[Type, Any]:
    """
    Convert a string representation of a Python literal into its corresponding type.
    Returns a tuple of (type, parsed_value).
    """
    try:
        json_input = json.loads(input_str) # Check if the input is valid JSON
        return (type(json_input), json_input)
    except json.JSONDecodeError:
        try:
            # If not JSON, check if if it is a list
            literal_input = ast.literal_eval(input_str)
            return (type(literal_input), literal_input)
        except (ValueError, SyntaxError):
            # If all else fails, return as string
            return (str, input_str)

def validate_input_type(input_type: Any, expected_type: str) -> bool:
    """
    Validate if the input value matches the expected type.
    Returns True if it matches, False otherwise.
    """
    exp = expected_type.lower().split()
    if "json" in exp:
        if input_type.__name__.lower() == "dict":
            return True
    elif "list" in exp:
        if input_type.__name__.lower() == "list":
            return True
    elif "string" in exp or "str" in exp:
        if input_type.__name__.lower() == "str":
            return True
    elif "int" in exp:
        if input_type.__name__.lower() == "int":
            return True
    elif "bool" in exp or "boolean" in exp:
        if input_type.__name__.lower() == "bool":
            return True
    return False

def lean_code_cleanup(lean_code: str, elaborate: bool = False) -> str:
    """
    Cleans up the error texts in the lean code.
    """
    final_code = []
    keywords_to_remove = ["#check", "trace", "Error: codegen:"]
    keywords_to_remove += ["import"] if elaborate else []
    for line in lean_code.splitlines():
        if not any(keyword in line for keyword in keywords_to_remove):
            final_code.append(line)

    if elaborate:
        return "\n".join(final_code).strip()
    return "import Mathlib\n" + "\n".join(final_code) if "import Mathlib" not in lean_code else "\n".join(final_code)

def process_lookup_response(lookup_response):
    """
    Process the lookup response from the server.
    0 : Job is completed by the server(what the response is, success or error, is independent)
    1 : Job is still running
    2 : Error in lookup
    """
    lookup_status = lookup_response.get("status", {})
    lookup_result = lookup_response.get("result", "error")

    if lookup_result != "success":
        # wrong token or similar
        return 2, lookup_response

    # This is for when job was successfully submitted
    if "completed" in lookup_status.keys():
        return 0, lookup_status["completed"]
    elif "running" in lookup_status:
        return 1, lookup_status["running"]
    else:
        return 3, lookup_response

def store_token_responses(token: str, status: str):
    """
    Store the token responses in session state.
    Also saves a timestamp of when it was last updated.
    """
    if not int(token) and int(token) > 0:
        return
    # Ensure the directory exists
    from datetime import datetime

    timestamp = datetime.now().strftime("%H:%M:%S %Y-%m-%d")
    if os.path.exists(TOKEN_JSON_FILE):
        with open(TOKEN_JSON_FILE, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        token_data[token] = {"status": status, "last_updated": timestamp}
        with open(TOKEN_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(token_data, f, indent=4)
    else:
        token_data = {token: {"status": status, "last_updated": timestamp}}
        with open(TOKEN_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(token_data, f, indent=4)
