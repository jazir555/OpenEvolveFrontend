#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick verification test for LoongFlow API server import fix.
Tests that the server can start and respond to health checks.
"""

import os
import subprocess
import sys
import time
import signal
import requests
from pathlib import Path

def test_server_startup():
    """Test that the API server starts without import errors."""

    print("=" * 70)
    print("LoongFlow API Server Import Fix - Verification Test")
    print("=" * 70)
    print()

    # Set required environment variable
    os.environ["LOONGFLOW_LLM_API_KEY"] = "sk-test-key-for-validation"

    # Start the server in a subprocess
    server_path = Path(__file__).parent / "api_server.py"

    print(f"🚀 Starting server from: {server_path}")
    print()

    try:
        # Start server process
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        print("⏳ Waiting for server to start...")

        # Wait for server to initialize
        time.sleep(3)

        # Check if process is still running
        if process.poll() is not None:
            print("❌ FAILED: Server process exited unexpectedly")
            print()
            print("Output:")
            print(process.stdout.read())
            return False

        print("✅ Server process started successfully")
        print()

        # Test health endpoint
        print("🔍 Testing health endpoint...")
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print("✅ Health endpoint responded successfully")
                print()
                print("Response:")
                print(f"  Status: {data.get('status')}")
                print(f"  Service: {data.get('service')}")
                print(f"  Version: {data.get('version')}")
                print(f"  Timestamp: {data.get('timestamp')}")
                print()
                return True
            else:
                print(f"❌ Health endpoint returned status {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print("❌ FAILED: Could not connect to server")
            print("   The server may not be listening on port 8000")
            return False
        except Exception as e:
            print(f"❌ FAILED: Error connecting to server: {e}")
            return False

    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False

    finally:
        # Clean up: terminate server process
        if 'process' in locals():
            print("🛑 Stopping server...")
            try:
                process.terminate()
                process.wait(timeout=5)
                print("✅ Server stopped")
            except Exception as e:
                print(f"⚠️  Warning: Could not stop server gracefully: {e}")
                try:
                    process.kill()
                except:
                    pass

        print()
        print("=" * 70)
        print("Test Complete")
        print("=" * 70)

if __name__ == "__main__":
    success = test_server_startup()
    sys.exit(0 if success else 1)
