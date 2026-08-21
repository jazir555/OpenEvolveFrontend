from __future__ import annotations

import PyPDF2
import os
import sys

def read_pdf(file_path):
    """Read and extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += extracted_text
                print(f"Page {page_num + 1}: Extracted {len(extracted_text)} characters")
                
            return text
    except Exception as e:
        error_msg = f"Error reading PDF: {str(e)}"
        print(error_msg)
        return error_msg

# Path to the PDF file
pdf_path = r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\smart contracts\2024-07-04_Quantstamp_v1.2.0.pdf"

# Extract the PDF content
print("Reading PDF file...")
pdf_content = read_pdf(pdf_path)

# Write to a file with error handling
try:
    with open("pdf_extract.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write(pdf_content)
    print("PDF content successfully written to pdf_extract.txt")
    
    # Also print length to confirm content
    print(f"Total characters extracted: {len(pdf_content)}")
    
except Exception as e:
    print(f"Error writing file: {str(e)}")