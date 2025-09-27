# Read the file
with open('mainlayout.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Create a new list of lines without duplicates
new_lines = []
skip_lines = False

for line in lines:
    # Skip duplicated content
    if 'st.success("Successfully committed to GitHub!")' in line:
        skip_lines = True
    elif 'else:' in line and skip_lines:
        skip_lines = False
        continue  # Skip the 'else:' line as well
    elif not skip_lines:
        new_lines.append(line)

# Write the cleaned content back to the file
with open('mainlayout.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)