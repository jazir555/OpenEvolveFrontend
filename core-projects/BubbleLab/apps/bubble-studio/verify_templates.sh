#!/bin/bash
# Quick verification script for both templates

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Template Verification Script                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "📋 Checking template files..."
echo ""

# Check if template files exist
if [ -f "src/components/templates/template_codes/websiteLeadGeneration.ts" ]; then
    echo -e "${GREEN}✓${NC} websiteLeadGeneration.ts exists"
else
    echo -e "${RED}✗${NC} websiteLeadGeneration.ts NOT FOUND"
fi

if [ -f "src/components/templates/template_codes/nanobananaImagePipeline.ts" ]; then
    echo -e "${GREEN}✓${NC} nanobananaImagePipeline.ts exists"
else
    echo -e "${RED}✗${NC} nanobananaImagePipeline.ts NOT FOUND"
fi

echo ""
echo "📋 Checking template registration..."
echo ""

# Check if templates are registered in templateLoader
if grep -q "website-lead-gen" src/components/templates/templateLoader.ts; then
    echo -e "${GREEN}✓${NC} website-lead-gen registered in templateLoader"
else
    echo -e "${RED}✗${NC} website-lead-gen NOT registered"
fi

if grep -q "nanobanana-image-pipeline" src/components/templates/templateLoader.ts; then
    echo -e "${GREEN}✓${NC} nanobanana-image-pipeline registered in templateLoader"
else
    echo -e "${RED}✗${NC} nanobanana-image-pipeline NOT registered"
fi

echo ""
echo "📋 Checking imports..."
echo ""

if grep -q "websiteLeadGenTemplate" src/components/templates/templateLoader.ts; then
    echo -e "${GREEN}✓${NC} websiteLeadGenTemplate imported"
else
    echo -e "${RED}✗${NC} websiteLeadGenTemplate NOT imported"
fi

if grep -q "nanobananaImagePipelineTemplate" src/components/templates/templateLoader.ts; then
    echo -e "${GREEN}✓${NC} nanobananaImagePipelineTemplate imported"
else
    echo -e "${RED}✗${NC} nanobananaImagePipelineTemplate NOT imported"
fi

echo ""
echo "📋 Template structure validation..."
echo ""

# Check websiteLeadGeneration structure
if grep -q "export const templateCode" src/components/templates/template_codes/websiteLeadGeneration.ts && \
   grep -q "export const metadata" src/components/templates/template_codes/websiteLeadGeneration.ts && \
   grep -q "extends BubbleFlow" src/components/templates/template_codes/websiteLeadGeneration.ts; then
    echo -e "${GREEN}✓${NC} websiteLeadGeneration structure valid"
else
    echo -e "${RED}✗${NC} websiteLeadGeneration structure invalid"
fi

# Check nanobananaImagePipeline structure
if grep -q "export const templateCode" src/components/templates/template_codes/nanobananaImagePipeline.ts && \
   grep -q "export const metadata" src/components/templates/template_codes/nanobananaImagePipeline.ts && \
   grep -q "extends BubbleFlow" src/components/templates/template_codes/nanobananaImagePipeline.ts; then
    echo -e "${GREEN}✓${NC} nanobananaImagePipeline structure valid"
else
    echo -e "${RED}✗${NC} nanobananaImagePipeline structure invalid"
fi

echo ""
echo "📋 Counting templates in templateLoader..."
TOTAL=$(grep -c "id: '" src/components/templates/templateLoader.ts || echo "0")
echo -e "${BLUE}ℹ${NC} Total templates registered: $TOTAL"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Verification Complete                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "To run full tests:"
echo "  1. Node.js validation: node test_templates_validation.cjs"
echo "  2. TypeScript test: npx tsx test_template_instantiation.ts"
echo ""
echo "To use templates:"
echo "  1. Start Bubble Studio"
echo "  2. Click 'Create from Template'"
echo "  3. Select 'Website Lead Generation' or 'Nanobanana Image Pipeline'"
echo ""
