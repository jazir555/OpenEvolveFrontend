#!/usr/bin/env python3
"""BubbleLab startup configuration validation.

Loads the YAML environment files and verifies they are safe to ship:
  (a) required security-critical environment variables are referenced;
  (b) no literal `devpassword`, bare example domains, or obvious hardcoded
      credentials remain outside environment-variable placeholders.

Uses the Python stdlib `yaml` when importable; otherwise falls back to a
simple text/grep scan so the script is dependency-light.

Run with:  python scripts/validate_config.py
"""

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, "config", "environments")
FILES = ["dev.yaml", "staging.yaml"]

# Security-critical env vars that MUST be referenced in the env files.
REQUIRED_VARS = [
    "KNOWLEDGE_GRAPH_DATABASE_URL",
    "ANALYTICS_DATABASE_URL",
    "REDIS_URL",
    "DATABASE_URL",
    "GOOGLE_OAUTH_CLIENT_SECRET",
]

# Forbidden literal secrets (never commit a real secret).
FORBIDDEN = ["devpassword"]

# Credential embedded in a URL, e.g. postgresql://user:secret@host
CREDENTIAL_RE = re.compile(r"://[^/\s:@]+:[^/\s:@]+@")

problems = []


def reference(var, text):
    """True if ${VAR} (with optional default) appears in text."""
    return "${%s" % var in text


try:
    import yaml  # type: ignore

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def scan_file(name):
    path = os.path.join(ENV_DIR, name)
    if not os.path.isfile(path):
        problems.append("missing file: %s" % path)
        return
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    # (1) Must parse as valid YAML when yaml is available.
    if HAVE_YAML:
        try:
            yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            problems.append("%s: YAML parse error: %s" % (name, exc))
            return

    # (2) No forbidden literal secrets, anywhere.
    for bad in FORBIDDEN:
        if bad in text:
            problems.append("%s: forbidden literal '%s' present" % (name, bad))

    # (3) Example domains must only appear inside ${...} placeholders
    #     (ignore comment lines, which may mention them for documentation).
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "example.com" in line and "${" not in line:
            problems.append(
                "%s: bare example domain (not env-var wrapped): %s"
                % (name, stripped)
            )

    # (4) Obvious hardcoded credentials must be wrapped in ${...}.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if CREDENTIAL_RE.search(line) and "${" not in line:
            problems.append(
                "%s: hardcoded credential outside env-var placeholder: %s"
                % (name, stripped)
            )

    # (5) Required env vars must be referenced.
    for var in REQUIRED_VARS:
        if not reference(var, text):
            problems.append(
                "%s: required env var '%s' not referenced" % (name, var)
            )


def main():
    for f in FILES:
        scan_file(f)

    if problems:
        print("CONFIG VALIDATION FAILED")
        for p in problems:
            print(" - " + p)
        sys.exit(1)

    print("CONFIG VALIDATION PASSED")
    print(" - parsed with yaml: %s" % HAVE_YAML)
    print(" - files checked: %s" % ", ".join(FILES))
    print(" - no forbidden secrets (e.g. devpassword)")
    print(" - example domains only inside ${VAR} placeholders")
    print(" - required env vars referenced")


if __name__ == "__main__":
    main()
