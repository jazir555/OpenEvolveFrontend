#!/usr/bin/env python3
"""
Check for truly redundant files that have identical or nearly identical content
"""

import os
import hashlib

def get_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def find_similar_files(directory):
    """Find files with identical content."""
    file_hashes = {}
    base_name_groups = {}

    # Group files by their base name (without suffixes)
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            # Remove common suffixes to get base name
            base_name = filename.replace('_FIXED.py', '.py').replace('_fixed.py', '.py').replace('_backup.py', '.py').replace('_backup_fix.py', '.py').replace('_EDGE_CASE_FIXES.py', '.py').replace('_backup2.py', '.py').replace('_original.py', '.py').replace('_v2.py', '.py').replace('_new.py', '.py').replace('_refactored.py', '.py').replace('_improved.py', '.py').replace('_enhanced.py', '.py').replace('_optimized.py', '.py').replace('_updated.py', '.py').replace('_final.py', '.py').replace('_legacy.py', '.py').replace('_old.py', '.py').replace('_deprecated.py', '.py').replace('_temp.py', '.py').replace('_copy.py', '.py').replace('_duplicate.py', '.py').replace('_simple.py', '.py').replace('_basic.py', '.py').replace('_minimal.py', '.py').replace('_lite.py', '.py').replace('_core.py', '.py').replace('_main.py', '.py').replace('_primary.py', '.py').replace('_standard.py', '.py').replace('_default.py', '.py').replace('_complete.py', '.py').replace('_full.py', '.py').replace('_comprehensive.py', '.py').replace('_advanced.py', '.py').replace('_pro.py', '.py').replace('_test.py', '.py').replace('_demo.py', '.py').replace('_example.py', '.py').replace('_sample.py', '.py').replace('_stub.py', '.py').replace('_mock.py', '.py').replace('_sim.py', '.py').replace('_prototype.py', '.py').replace('_draft.py', '.py').replace('_template.py', '.py').replace('_skeleton.py', '.py')

            filepath = os.path.join(directory, filename)
            file_hash = get_file_hash(filepath)

            # Group by hash
            if file_hash not in file_hashes:
                file_hashes[file_hash] = []
            file_hashes[file_hash].append(filename)

            # Also group by base name
            if base_name not in base_name_groups:
                base_name_groups[base_name] = []
            base_name_groups[base_name].append(filename)

    # Print files with identical content
    print("Files with IDENTICAL content:")
    print("="*50)
    identical_found = False
    for hash_val, files in file_hashes.items():
        if len(files) > 1:
            identical_found = True
            print(f"Hash: {hash_val[:16]}...")
            for file in files:
                print(f"  -> {file}")
            print()

    if not identical_found:
        print("No files with identical content found.")

    # Print files with same base name (potential duplicates)
    print("\nFiles with SAME BASE NAME (potential duplicates to review):")
    print("="*50)
    base_duplicates_found = False
    for base_name, files in base_name_groups.items():
        if len(files) > 1:
            base_duplicates_found = True
            print(f"Base name: {base_name}")
            for file in files:
                filepath = os.path.join(directory, file)
                size = os.path.getsize(filepath)
                print(f"  -> {file} ({size} bytes)")
            print()

    if not base_duplicates_found:
        print("No files with same base name found.")

    return file_hashes, base_name_groups

if __name__ == "__main__":
    directory = "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/"
    find_similar_files(directory)