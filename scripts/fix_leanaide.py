with open('leanaide_mdap_demo.py', 'r') as f:
    content = f.read()

# Fix all instances of split print statements
# Pattern: print('\n' + '.....')
content = content.replace("print('\n'\n    '+'", "print(")
content = content.replace("print('\n' +", "print(")

with open('leanaide_mdap_demo.py', 'w') as f:
    f.write(content)

print("Fixed")
