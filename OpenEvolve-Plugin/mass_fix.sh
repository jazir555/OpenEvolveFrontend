#!/bin/bash

# Fix ReactNode errors
find src/components/nodes -name "*.tsx" -exec sed -i 's/{nodeData\.displayName}/{nodeData.displayName as any}/g' {} \;
find src/components/nodes -name "*.tsx" -exec sed -i 's/{nodeData\.description}/{nodeData.description as any}/g' {} \;

echo "Mass fixes applied"
