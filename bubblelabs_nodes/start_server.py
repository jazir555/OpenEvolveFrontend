"""
OpenEvolve BubbleLabs API Server - Development Startup Script

This script starts the FastAPI server with proper initialization,
dependency checking, and informative console output.
"""

import uvicorn
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('websockets', 'WebSockets'),
        ('pydantic', 'Pydantic'),
    ]

    missing = []
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT FOUND")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall them with:")
        print("  pip install -r requirements.txt")
        return False

    print("✓ All dependencies satisfied\n")
    return True


def check_openevolve_integration():
    """Check if OpenEvolve components are available."""
    print("Checking OpenEvolve integration...")

    try:
        from bubblelabs_nodes import NodeRegistry, list_nodes
        nodes = list_nodes()
        print(f"  ✓ NodeRegistry loaded")
        print(f"  ✓ Found {len(nodes)} registered nodes")
        return True
    except ImportError as e:
        print(f"  ⚠ NodeRegistry not available: {e}")
        print("  Continuing anyway...\n")
        return True


def show_registered_nodes():
    """Display all registered nodes with their details."""
    try:
        from bubblelabs_nodes import list_nodes
        nodes = list_nodes()

        print("=" * 60)
        print(f"REGISTERED NODES ({len(nodes)})")
        print("=" * 60)

        for node_type, node_class in sorted(nodes.items()):
            display_name = getattr(node_class, 'DISPLAY_NAME', node_type)
            description = getattr(node_class, 'DESCRIPTION', 'No description')
            print(f"\n{node_type}")
            print(f"  Name: {display_name}")
            print(f"  Description: {description}")

        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"⚠ Could not list nodes: {e}\n")


def show_api_info():
    """Display API server information."""
    print("=" * 60)
    print("OPENEVOLVE BUBBLELABS API SERVER")
    print("=" * 60)
    print(f"Version: 1.0.0")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 60)
    print("\nServer Information:")
    print("  URL: http://localhost:8000")
    print("  API Docs: http://localhost:8000/docs")
    print("  ReDoc: http://localhost:8000/redoc")
    print("  Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")


def main():
    """Main entry point for starting the server."""
    print("\n" + "=" * 60)
    print("STARTING OPENEVOLVE BUBBLELABS API SERVER")
    print("=" * 60 + "\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check OpenEvolve integration
    check_openevolve_integration()

    # Import FastAPI app
    try:
        from api_server import app
        print("✓ FastAPI application loaded\n")
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        print("\nMake sure api_server.py exists in the current directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading FastAPI app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Show registered nodes
    show_registered_nodes()

    # Show API information
    show_api_info()

    # Configure uvicorn
    config = {
        "app": app,
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,  # Enable auto-reload for development
        "log_level": "info",
        "access_log": True,
    }

    # Check for production flag
    if "--production" in sys.argv or "-p" in sys.argv:
        config["reload"] = False
        config["log_level"] = "warning"
        print("🔧 Running in PRODUCTION mode (auto-reload disabled)\n")

    # Start server
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("SERVER SHUTTING DOWN")
        print("=" * 60)
        print("\n✓ Server stopped gracefully")
    except SystemExit:
        print("\n\n" + "=" * 60)
        print("SERVER EXITING")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
