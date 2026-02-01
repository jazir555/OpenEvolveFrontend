#!/bin/bash
#
# OpenEvolve gRPC Code Generation Script
#
# Generates TypeScript and Python code from protobuf definitions
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"
PYTHON_OUT="$PROJECT_ROOT/python/generated"
TYPESCRIPT_OUT="$PROJECT_ROOT/typescript/generated"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}OpenEvolve gRPC Code Generation${NC}"
echo "================================"
echo ""

# Check for required tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

echo "Checking prerequisites..."
check_command protoc
check_command python3
check_command npm
echo -e "${GREEN}All prerequisites found${NC}"
echo ""

# Create output directories
mkdir -p "$PYTHON_OUT"
mkdir -p "$TYPESCRIPT_OUT"

# Find all proto files
PROTO_FILES=$(find "$PROTO_DIR" -name "*.proto" -type f | sort)

echo "Found proto files:"
echo "$PROTO_FILES"
echo ""

# ============================================================================
# Generate Python Code
# ============================================================================

echo -e "${YELLOW}Generating Python code...${NC}"

# Install grpcio-tools if needed
if ! python3 -c "import grpc_tools" 2>/dev/null; then
    echo "Installing grpcio-tools..."
    pip3 install grpcio-tools
fi

# Generate Python gRPC code
python3 -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$PYTHON_OUT" \
    --grpc_python_out="$PYTHON_OUT" \
    --pyi_out="$PYTHON_OUT" \
    $PROTO_FILES

# Fix Python imports (workaround for protobuf import issue)
echo "Fixing Python imports..."
for f in "$PYTHON_OUT"/*_pb2*.py; do
    if [ -f "$f" ]; then
        # Replace 'import xxx_pb2' with 'from . import xxx_pb2'
        sed -i 's/^import \([^ ]*_pb2\)/from . import \1/g' "$f" 2>/dev/null || true
    fi
done

echo -e "${GREEN}Python code generated in $PYTHON_OUT${NC}"
echo ""

# ============================================================================
# Generate TypeScript Code
# ============================================================================

echo -e "${YELLOW}Generating TypeScript code...${NC}"

# Install grpc-tools if needed
if ! npm list -g grpc-tools &>/dev/null; then
    echo "Installing grpc-tools..."
    npm install -g grpc-tools
fi

if ! npm list -g grpc_tools_node_protoc_ts &>/dev/null; then
    echo "Installing grpc_tools_node_protoc_ts..."
    npm install -g grpc_tools_node_protoc_ts
fi

# Generate TypeScript gRPC code
for proto_file in $PROTO_FILES; do
    echo "Processing: $(basename "$proto_file")"
    
    grpc_tools_node_protoc \
        --js_out="import_style=commonjs,binary:$TYPESCRIPT_OUT" \
        --grpc_out="grpc_js:$TYPESCRIPT_OUT" \
        --plugin="protoc-gen-grpc=$(which grpc_tools_node_protoc_plugin)" \
        --proto_path="$PROTO_DIR" \
        "$proto_file"
    
    # Generate TypeScript definitions
    grpc_tools_node_protoc \
        --plugin="protoc-gen-ts=$(which protoc-gen-ts)" \
        --ts_out="$TYPESCRIPT_OUT" \
        --proto_path="$PROTO_DIR" \
        "$proto_file"
done

echo -e "${GREEN}TypeScript code generated in $TYPESCRIPT_OUT${NC}"
echo ""

# ============================================================================
# Generate Documentation
# ============================================================================

echo -e "${YELLOW}Generating documentation...${NC}"

# Install protoc-gen-doc if needed
if ! command -v protoc-gen-doc &>/dev/null; then
    echo "Installing protoc-gen-doc..."
    if command -v go &>/dev/null; then
        go install github.com/pseudomuto/protoc-gen-doc/cmd/protoc-gen-doc@latest
    else
        echo -e "${YELLOW}Warning: Go not installed, skipping doc generation${NC}"
    fi
fi

if command -v protoc-gen-doc &>/dev/null; then
    mkdir -p "$PROJECT_ROOT/docs"
    
    protoc \
        --doc_out="$PROJECT_ROOT/docs" \
        --doc_opt=markdown,api.md \
        --proto_path="$PROTO_DIR" \
        $PROTO_FILES
    
    echo -e "${GREEN}Documentation generated in $PROJECT_ROOT/docs${NC}"
else
    echo -e "${YELLOW}Skipping documentation generation${NC}"
fi

echo ""
echo -e "${GREEN}Code generation complete!${NC}"
echo ""
echo "Output locations:"
echo "  Python: $PYTHON_OUT"
echo "  TypeScript: $TYPESCRIPT_OUT"
echo "  Docs: $PROJECT_ROOT/docs"
