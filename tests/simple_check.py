# Simple approach - look for the problematic triple quote patterns manually
with open('mainlayout_fixed_newlines.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Count \" and \"\"\" occurrences to understand the structure
quote_count = content.count('\"')
triple_quote_count = content.count('\"\"\"')

print(f'Single quotes: {quote_count}')
print(f'Triple quotes: {triple_quote_count}')

# Look for the pattern around the sample protocol specifically
sample_start = content.find('sample_protocol = \"\"\"# Sample Security Policy')
if sample_start != -1:
    print(f'Found sample protocol at position: {sample_start}')
    # Look for the end of this string in the next few thousand characters
    sample_end = content.find('\"\"\"', sample_start + 20)
    next_occurrence = content.find('\"\"\"', sample_end + 3)
    print(f'Next triple quote after sample start: {next_occurrence}')
    
    # Get a section around the sample protocol
    start = max(0, sample_start - 100)
    end = min(len(content), sample_start + 1000)
    section = content[start:end]
    print(f'Section around sample protocol: {repr(section[:300])}...')
else:
    print('Sample protocol not found with exact text')
