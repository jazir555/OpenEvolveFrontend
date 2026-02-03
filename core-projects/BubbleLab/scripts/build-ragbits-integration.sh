#!/bin/bash
# Build script for ragbits-bubblelab-integration

echo "Building ragbits-bubblelab-integration..."

# Navigate to the package directory
cd "$(dirname "$0")/../packages/ragbits-bubblelab-integration"

# Run the build
npm run build

echo "Build completed!"