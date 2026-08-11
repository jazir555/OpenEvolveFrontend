with open('pdf_extract.txt', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
    print(content[-2000:])