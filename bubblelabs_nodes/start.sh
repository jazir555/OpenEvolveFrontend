#!/bin/bash
###############################################################################
# OpenEvolve BubbleLabs Backend - Linux/macOS Startup Script
#
# This script starts the Python backend server with proper environment setup
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OPENEVOLVE BUBBLELABS BACKEND STARTUP${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment found${NC}"
    echo "It's recommended to create one with:"
    echo "  python3 -m venv venv"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Checking dependencies...${NC}"

    # Install dependencies if not already installed
    if ! python3 -c "import fastapi" &> /dev/null; then
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install -q -r requirements.txt
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    else
        echo -e "${GREEN}✓ Dependencies already installed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}STARTING SERVER${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Start the server
python3 start_server.py

# Exit cleanup
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Server stopped successfully${NC}"
else
    echo -e "${RED}✗ Server exited with code ${EXIT_CODE}${NC}"
fi

exit $EXIT_CODE
