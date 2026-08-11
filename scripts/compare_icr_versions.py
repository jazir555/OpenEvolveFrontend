"""
ICR Directory Comparison Script
Compares local vs upstream ICR directories
"""
import os
from pathlib import Path
from datetime import datetime

local_dir = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\core-projects\Iterative-Contextual-Refinements")
upstream_dir = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\core-projects\Iterative-Contextual-Refinements-main")

def get_all_files(base_dir):
    """Get all files with relative paths and sizes"""
    files = {}
    for root, dirs, filenames in os.walk(base_dir):
        # Skip node_modules
        if 'node_modules' in root:
            continue
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(base_dir)
            try:
                size = full_path.stat().st_size
                mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
                files[str(rel_path)] = {'size': size, 'mtime': mtime, 'full_path': full_path}
            except Exception as e:
                print(f"Error accessing {full_path}: {e}")
    return files

print("=" * 80)
print("ICR DIRECTORY COMPARISON")
print("=" * 80)
print(f"\nLocal: {local_dir}")
print(f"Upstream: {upstream_dir}")

print("\n" + "=" * 80)
print("COLLECTING FILES...")
print("=" * 80)

local_files = get_all_files(local_dir)
upstream_files = get_all_files(upstream_dir)

print(f"\nLocal files: {len(local_files)}")
print(f"Upstream files: {len(upstream_files)}")

# Calculate total size
local_size = sum(f['size'] for f in local_files.values())
upstream_size = sum(f['size'] for f in upstream_files.values())

print(f"\nLocal total size: {local_size / 1024 / 1024:.2f} MB")
print(f"Upstream total size: {upstream_size / 1024 / 1024:.2f} MB")

# Find unique and common files
local_only = set(local_files.keys()) - set(upstream_files.keys())
upstream_only = set(upstream_files.keys()) - set(local_files.keys())
common = set(local_files.keys()) & set(upstream_files.keys())

print("\n" + "=" * 80)
print("FILE COMPARISON SUMMARY")
print("=" * 80)
print(f"\nFiles ONLY in LOCAL: {len(local_only)}")
print(f"Files ONLY in UPSTREAM: {len(upstream_only)}")
print(f"Files in BOTH: {len(common)}")

# Analyze by directory
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
print("LOCAL-ONLY FILES BY DIRECTORY")
print("=" * 80)
local_by_dir = group_by_directory(local_only)
for dir_name, files in sorted(local_by_dir.items()):
    print(f"\n{dir_name}/ ({len(files)} files)")
    for f in sorted(files)[:10]:  # Show first 10
        size = local_files[f]['size']
        print(f"  - {f} ({size:,} bytes)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

print("\n" + "=" * 80)
print("UPSTREAM-ONLY FILES BY DIRECTORY")
print("=" * 80)
upstream_by_dir = group_by_directory(upstream_only)
for dir_name, files in sorted(upstream_by_dir.items()):
    print(f"\n{dir_name}/ ({len(files)} files)")
    for f in sorted(files)[:10]:  # Show first 10
        size = upstream_files[f]['size']
        print(f"  - {f} ({size:,} bytes)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

print("\n" + "=" * 80)
print("COMMON FILES (MODIFIED CHECK)")
print("=" * 80)
modified_files = []
for f in sorted(common):
    local_size = local_files[f]['size']
    upstream_size = upstream_files[f]['size']
    local_mtime = local_files[f]['mtime']
    upstream_mtime = upstream_files[f]['mtime']
    
    if local_size != upstream_size:
        modified_files.append({
            'path': f,
            'local_size': local_size,
            'upstream_size': upstream_size,
            'local_mtime': local_mtime,
            'upstream_mtime': upstream_mtime,
            'size_diff': local_size - upstream_size
        })

print(f"\nCommon files with SIZE DIFFERENCES: {len(modified_files)}")
print("\nTop 20 modified files (by size difference):")
modified_files.sort(key=lambda x: abs(x['size_diff']), reverse=True)
for f in modified_files[:20]:
    print(f"\n  {f['path']}")
    print(f"    Local: {f['local_size']:,} bytes ({f['local_mtime']})")
    print(f"    Upstream: {f['upstream_size']:,} bytes ({f['upstream_mtime']})")
    print(f"    Diff: {f['size_diff']:+,} bytes")

# Save full report
report_path = Path(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\docs\Iterative Contextual Refinements\ICR_DIRECTORY_COMPARISON_REPORT.md")

with open(report_path, 'w', encoding='utf-8') as report:
    report.write("# ICR Directory Comparison Report\n\n")
    report.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
    report.write(f"- **Local files:** {len(local_files)}\n")
    report.write(f"- **Upstream files:** {len(upstream_files)}\n")
    report.write(f"- **Local size:** {local_size / 1024 / 1024:.2f} MB\n")
    report.write(f"- **Upstream size:** {upstream_size / 1024 / 1024:.2f} MB\n\n")
    
    report.write("## Local-Only Files\n\n")
    report.write(f"Total: {len(local_only)} files\n\n")
    for dir_name, files in sorted(local_by_dir.items()):
        report.write(f"### {dir_name}/ ({len(files)} files)\n\n")
        for f in sorted(files):
            size = local_files[f]['size']
            report.write(f"- `{f}` ({size:,} bytes)\n")
        report.write("\n")
    
    report.write("## Upstream-Only Files\n\n")
    report.write(f"Total: {len(upstream_only)} files\n\n")
    for dir_name, files in sorted(upstream_by_dir.items()):
        report.write(f"### {dir_name}/ ({len(files)} files)\n\n")
        for f in sorted(files):
            size = upstream_files[f]['size']
            report.write(f"- `{f}` ({size:,} bytes)\n")
        report.write("\n")
    
    report.write("## Modified Common Files\n\n")
    report.write(f"Total: {len(modified_files)} files with size differences\n\n")
    for f in modified_files:
        report.write(f"### `{f['path']}`\n\n")
        report.write(f"- Local: {f['local_size']:,} bytes\n")
        report.write(f"- Upstream: {f['upstream_size']:,} bytes\n")
        report.write(f"- Difference: {f['size_diff']:+,} bytes\n\n")

print(f"\n" + "=" * 80)
print(f"FULL REPORT SAVED TO:")
print(f"{report_path}")
print("=" * 80)
