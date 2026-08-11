"""
ICR Gap Analysis Script
Compares upstream vs local to identify missing files
"""
import os
from pathlib import Path
from datetime import datetime

local_dir = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\core-projects\Iterative-Contextual-Refinements")
upstream_dir = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\core-projects\Iterative-Contextual-Refinements-main")

def get_all_files(base_dir):
    """Get all files with relative paths, sizes, and modification times"""
    files = {}
    for root, dirs, filenames in os.walk(base_dir):
        # Skip node_modules and .git
        if 'node_modules' in root or '.git' in root:
            continue
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(base_dir)
            try:
                stat = full_path.stat()
                files[str(rel_path)] = {
                    'size': stat.st_size,
                    'mtime': datetime.fromtimestamp(stat.st_mtime),
                    'full_path': full_path
                }
            except Exception as e:
                print(f"Error accessing {full_path}: {e}")
    return files

print("=" * 80)
print("ICR GAP ANALYSIS - COMPREHENSIVE COMPARISON")
print("=" * 80)
print(f"\nLocal: {local_dir}")
print(f"Upstream: {upstream_dir}")
print(f"\nAnalysis Date: {datetime.now().isoformat()}")

print("\n" + "=" * 80)
print("COLLECTING FILES...")
print("=" * 80)

local_files = get_all_files(local_dir)
upstream_files = get_all_files(upstream_dir)

print(f"\nLocal files: {len(local_files)}")
print(f"Upstream files: {len(upstream_files)}")

# Calculate total sizes
local_size = sum(f['size'] for f in local_files.values())
upstream_size = sum(f['size'] for f in upstream_files.values())

print(f"\nLocal total size: {local_size / 1024 / 1024:.2f} MB")
print(f"Upstream total size: {upstream_size / 1024 / 1024:.2f} MB")

# Find unique and common files
local_only = set(local_files.keys()) - set(upstream_files.keys())
upstream_only = set(upstream_files.keys()) - set(local_files.keys())
common = set(local_files.keys()) & set(upstream_files.keys())

print("\n" + "=" * 80)
print("GAP ANALYSIS SUMMARY")
print("=" * 80)
print(f"\nFiles ONLY in LOCAL (custom features): {len(local_only)}")
print(f"Files ONLY in UPSTREAM (to be ported): {len(upstream_only)}")
print(f"Files in BOTH (need comparison): {len(common)}")

# Analyze by directory for upstream-only files
def group_by_directory(file_set):
    """Group files by their top-level directory"""
    dirs = {}
    for f in file_set:
        parts = f.split('\\')
        top_dir = parts[0] if len(parts) > 1 else '(root)'
        if top_dir not in dirs:
            dirs[top_dir] = []
        dirs[top_dir].append(f)
    return dirs

print("\n" + "=" * 80)
print("UPSTREAM-ONLY FILES (NEED TO PORT)")
print("=" * 80)

if upstream_only:
    upstream_by_dir = group_by_directory(upstream_only)
    for dir_name, files in sorted(upstream_by_dir.items()):
        print(f"\n{dir_name}/ ({len(files)} files)")
        total_size = sum(upstream_files[f]['size'] for f in files)
        print(f"  Total size: {total_size / 1024:.1f} KB")
        for f in sorted(files):
            size = upstream_files[f]['size']
            mtime = upstream_files[f]['mtime'].strftime('%Y-%m-%d %H:%M')
            print(f"  - {f} ({size:,} bytes, {mtime})")
else:
    print("\n✅ NO UPSTREAM-ONLY FILES - ALL UPSTREAM FILES HAVE BEEN PORTED!")

print("\n" + "=" * 80)
print("LOCAL-ONLY FILES (CUSTOM FEATURES - PRESERVE)")
print("=" * 80)

local_by_dir = group_by_directory(local_only)
for dir_name, files in sorted(local_by_dir.items()):
    print(f"\n{dir_name}/ ({len(files)} files)")
    total_size = sum(local_files[f]['size'] for f in files)
    print(f"  Total size: {total_size / 1024:.1f} KB")
    for f in sorted(files)[:15]:  # Show first 15
        size = local_files[f]['size']
        print(f"  - {f} ({size:,} bytes)")
    if len(files) > 15:
        print(f"  ... and {len(files) - 15} more")

print("\n" + "=" * 80)
print("COMMON FILES WITH SIZE DIFFERENCES")
print("=" * 80)

modified_files = []
for f in sorted(common):
    local_size = local_files[f]['size']
    upstream_size = upstream_files[f]['size']
    
    if local_size != upstream_size:
        size_diff = local_size - upstream_size
        size_diff_pct = (size_diff / upstream_size * 100) if upstream_size > 0 else 0
        modified_files.append({
            'path': f,
            'local_size': local_size,
            'upstream_size': upstream_size,
            'diff': size_diff,
            'diff_pct': size_diff_pct
        })

print(f"\nTotal modified files: {len(modified_files)}")

# Group by directory
modified_by_dir = {}
for f in modified_files:
    parts = f['path'].split('\\')
    top_dir = parts[0] if len(parts) > 1 else '(root)'
    if top_dir not in modified_by_dir:
        modified_by_dir[top_dir] = []
    modified_by_dir[top_dir].append(f)

for dir_name, files in sorted(modified_by_dir.items()):
    print(f"\n{dir_name}/ ({len(files)} files)")
    # Sort by absolute size difference
    files.sort(key=lambda x: abs(x['diff']), reverse=True)
    for f in files[:10]:  # Show top 10
        sign = '+' if f['diff'] > 0 else ''
        print(f"  {f['path']}")
        print(f"    Local: {f['local_size']:,} bytes")
        print(f"    Upstream: {f['upstream_size']:,} bytes")
        print(f"    Diff: {sign}{f['diff']:,} bytes ({sign}{f['diff_pct']:.1f}%)")

print("\n" + "=" * 80)
print("GAP ANALYSIS CONCLUSIONS")
print("=" * 80)

if not upstream_only:
    print("\n✅ ALL UPSTREAM FILES HAVE BEEN PORTED!")
    print("\nNo additional files need to be ported from upstream.")
    print("\nLocal version contains:")
    print(f"  - All {len(upstream_files)} upstream files")
    print(f"  - {len(local_only)} additional custom files")
    print(f"  - Total: {len(local_files)} files ({local_size / 1024 / 1024:.2f} MB)")
else:
    print(f"\n⚠️  {len(upstream_only)} upstream files still need to be ported:")
    for dir_name, files in sorted(group_by_directory(upstream_only).items()):
        print(f"  - {dir_name}/: {len(files)} files")

print("\n" + "=" * 80)

# Save full report
report_path = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\docs\Iterative Contextual Refinements\ICR_GAP_ANALYSIS_REPORT.md")

with open(report_path, 'w', encoding='utf-8') as report:
    report.write("# ICR Gap Analysis Report\n\n")
    report.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
    report.write(f"- **Local files:** {len(local_files)}\n")
    report.write(f"- **Upstream files:** {len(upstream_files)}\n")
    report.write(f"- **Local size:** {local_size / 1024 / 1024:.2f} MB\n")
    report.write(f"- **Upstream size:** {upstream_size / 1024 / 1024:.2f} MB\n\n")
    
    if upstream_only:
        report.write("## Upstream-Only Files (To Port)\n\n")
        report.write(f"Total: {len(upstream_only)} files\n\n")
        upstream_by_dir = group_by_directory(upstream_only)
        for dir_name, files in sorted(upstream_by_dir.items()):
            report.write(f"### {dir_name}/ ({len(files)} files)\n\n")
            for f in sorted(files):
                size = upstream_files[f]['size']
                mtime = upstream_files[f]['mtime'].strftime('%Y-%m-%d %H:%M')
                report.write(f"- `{f}` ({size:,} bytes, {mtime})\n")
            report.write("\n")
    else:
        report.write("## ✅ ALL UPSTREAM FILES PORTED\n\n")
        report.write("No upstream-only files found. All upstream files have been successfully ported.\n\n")
    
    report.write("## Local-Only Files (Custom Features)\n\n")
    report.write(f"Total: {len(local_only)} files\n\n")
    for dir_name, files in sorted(local_by_dir.items()):
        report.write(f"### {dir_name}/ ({len(files)} files)\n\n")
        for f in sorted(files):
            size = local_files[f]['size']
            report.write(f"- `{f}` ({size:,} bytes)\n")
        report.write("\n")
    
    report.write("## Modified Common Files\n\n")
    report.write(f"Total: {len(modified_files)} files with size differences\n\n")
    for f in modified_files[:50]:  # Top 50
        sign = '+' if f['diff'] > 0 else ''
        report.write(f"### `{f['path']}`\n\n")
        report.write(f"- Local: {f['local_size']:,} bytes\n")
        report.write(f"- Upstream: {f['upstream_size']:,} bytes\n")
        report.write(f"- Difference: {sign}{f['diff']:,} bytes ({sign}{f['diff_pct']:.1f}%)\n\n")

print(f"FULL REPORT SAVED TO:")
print(f"{report_path}")
print("=" * 80)
