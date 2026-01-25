#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🗑️  Decommissioning Streamlit UI"
echo "================================"
echo ""

# Safety confirmation
read -p "Have you verified that BubbleLab UI is fully functional? (yes/no): " confirmation
if [ "$confirmation" != "yes" ]; then
    echo "Decommission aborted. Please verify BubbleLab UI first."
    exit 0
fi

read -p "Are you sure you want to decommission Streamlit? (yes/no): " confirmation2
if [ "$confirmation2" != "yes" ]; then
    echo "Decommission aborted."
    exit 0
fi

# Create archive directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="deprecated/streamlit_${TIMESTAMP}"
mkdir -p "$ARCHIVE_DIR"

echo "📁 Archive directory: $ARCHIVE_DIR"
echo ""

# Step 1: Find all Streamlit files
echo "Step 1: Finding all Streamlit files..."
streamlit_files=$(find . -type f -name "*.py" -exec grep -l "import streamlit\|import st\|from streamlit" {} \; 2>/dev/null || true)

if [ -z "$streamlit_files" ]; then
    echo -e "${YELLOW}⚠️  No Streamlit files found${NC}"
    exit 0
fi

file_count=$(echo "$streamlit_files" | wc -l)
echo -e "${GREEN}Found $file_count Streamlit files${NC}"
echo ""

# Step 2: Move files to archive
echo "Step 2: Archiving Streamlit files..."
moved_count=0
while IFS= read -r file; do
    if [ -f "$file" ]; then
        # Create directory structure in archive
        target_dir="$ARCHIVE_DIR/$(dirname "$file")"
        mkdir -p "$target_dir"

        # Move file
        git mv "$file" "$target_dir/" 2>/dev/null || mv "$file" "$target_dir/"
        ((moved_count++))
        echo "  Archived: $file"
    fi
done <<< "$streamlit_files"

echo -e "${GREEN}✅ Archived $moved_count files${NC}"
echo ""

# Step 3: Update requirements.txt
echo "Step 3: Updating requirements.txt..."
if [ -f "requirements.txt" ]; then
    # Create backup
    cp requirements.txt requirements.txt.backup

    # Remove streamlit
    sed -i '/streamlit/d' requirements.txt

    echo -e "${GREEN}✅ Removed streamlit from requirements.txt${NC}"
    echo "  Backup saved as: requirements.txt.backup"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi
echo ""

# Step 4: Remove Streamlit-specific config files
echo "Step 4: Removing Streamlit configuration..."
if [ -f ".streamlit/config.toml" ]; then
    git mv .streamlit/config.toml "$ARCHIVE_DIR/.streamlit/" 2>/dev/null || mv .streamlit/config.toml "$ARCHIVE_DIR/.streamlit/"
    echo "  Archived: .streamlit/config.toml"
fi

if [ -f "streamlit_app.py" ]; then
    git mv streamlit_app.py "$ARCHIVE_DIR/" 2>/dev/null || mv streamlit_app.py "$ARCHIVE_DIR/"
    echo "  Archived: streamlit_app.py"
fi

echo -e "${GREEN}✅ Streamlit configuration archived${NC}"
echo ""

# Step 5: Create decommission manifest
echo "Step 5: Creating decommission manifest..."
{
    echo "Streamlit Decommission Report"
    echo "=============================="
    echo ""
    echo "Decommission Date: $(date)"
    echo "Archive Location: $ARCHIVE_DIR"
    echo ""
    echo "Files Archived: $moved_count"
    echo ""
    echo "Replaced By: BubbleLab React UI"
    echo "Location: BubbleLab/apps/bubble-studio"
    echo ""
    echo "Git Commit: $(git rev-parse HEAD)"
    echo ""
    echo "Files List:"
    find "$ARCHIVE_DIR" -type f | sort
} > "$ARCHIVE_DIR/DECOMMISSION_MANIFEST.txt"

echo -e "${GREEN}✅ Manifest created${NC}"
echo ""

# Step 6: Create README in archive
echo "Step 6: Creating archive README..."
{
    echo "# Streamlit Archive"
    echo ""
    echo "This directory contains Streamlit files that were decommissioned on $(date)."
    echo ""
    echo "## Migration Information"
    echo ""
    echo "Streamlit has been replaced with BubbleLab React UI."
    echo "New UI location: \`BubbleLab/apps/bubble-studio\`"
    echo ""
    echo "## Rollback (if needed)"
    echo ""
    echo "To restore Streamlit (if absolutely necessary):"
    echo '```bash'
    echo "git mv deprecated/streamlit_${TIMESTAMP}/* ."
    echo "echo 'streamlit==1.28.0' >> requirements.txt"
    echo "pip install -r requirements.txt"
    echo '```'
    echo ""
    echo "## Archive Contents"
    echo ""
    find "$ARCHIVE_DIR" -type f | sed 's|^'"$ARCHIVE_DIR"'||' | sort
} > "$ARCHIVE_DIR/README.md"

echo -e "${GREEN}✅ Archive README created${NC}"
echo ""

# Step 7: Commit changes
echo "Step 7: Committing changes..."
if command -v git &> /dev/null; then
    git add .
    git commit -m "chore: decommission Streamlit UI

Archived all Streamlit files to $ARCHIVE_DIR
Removed streamlit from requirements.txt
Streamlit has been replaced with BubbleLab React UI

Files archived: $moved_count
Archive location: $ARCHIVE_DIR

Related: #MIGRATION-1

Decommission manifest:
$(cat "$ARCHIVE_DIR/DECOMMISSION_MANIFEST.txt")"
    echo -e "${GREEN}✅ Changes committed${NC}"
else
    echo -e "${YELLOW}⚠️  Git not available, skipping commit${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ STREAMLIT DECOMMISSION COMPLETE${NC}"
echo ""
echo "📁 Archive location: $ARCHIVE_DIR"
echo "📊 Files archived: $moved_count"
echo ""
echo "📝 Next steps:"
echo "  1. Update documentation to remove Streamlit references"
echo "  2. Update deployment scripts"
echo "  3. Update README.md"
echo "  4. Notify team of deprecation"
echo ""
echo "🔍 To view archived files:"
echo "  cd $ARCHIVE_DIR && ls -la"
echo ""
echo "📋 To view decommission manifest:"
echo "  cat $ARCHIVE_DIR/DECOMMISSION_MANIFEST.txt"
echo ""

exit 0
