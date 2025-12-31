# Document Parsing Implementation Guide

## Overview

This document provides comprehensive technical details for the implementation of document parsing in the Knowledge Extraction Agent. The system supports two parsing methods (Fast Parsing and Docling Parsing) with two output formats each (Markdown and Structured).

## Architecture

### Parsing Methods

1. **Fast Parsing**: Uses PyMuPDF and python-docx libraries
2. **Docling Parsing**: Uses Docling library with CUDA acceleration support

### Output Formats

1. **Markdown Format**: Clean, readable text extraction
2. **Structured Format**: Preserves document layout, positioning, and structure

## Fast Parsing Implementation

### Core Classes

#### DocumentParser Class

```python
class DocumentParser:
    """Open-source document parser for PDF and Word documents"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.doc']
    
    def parse_document(self, file_path: str, use_markdown: bool = True) -> Dict[str, Any]:
        """Main parsing method with format selection"""
        # Implementation details below
```

### PDF Parsing

#### Markdown Format (Simple Text Extraction)

```python
def parse_pdf(self, file_path: str, use_markdown: bool = True) -> str:
    """Parse PDF file and extract text content"""
    try:
        doc = fitz.open(file_path)
        
        if use_markdown:
            # Simple text extraction (current behavior)
            text_content = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content += page.get_text()
                text_content += "\n\n"  # Add page separator
        
        doc.close()
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        return ""
```

#### Structured Format (Layout Preservation)

```python
def _extract_structured_pdf(self, doc) -> str:
    """Extract text from PDF preserving layout and structure"""
    text_parts = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Add page header
        text_parts.append(f"=== PAGE {page_num + 1} ===")
        
        # Extract text blocks with positioning
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if "lines" in block:
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        if span["text"].strip():
                            # Add positioning info for structured extraction
                            x, y = span["bbox"][0], span["bbox"][1]
                            line_text.append(f"[x:{x:.0f},y:{y:.0f}] {span['text']}")
                    if line_text:
                        block_text.append(" ".join(line_text))
                if block_text:
                    text_parts.append("\n".join(block_text))
        
        text_parts.append("")  # Page separator
    
    return "\n".join(text_parts)
```

### DOCX Parsing

#### Markdown Format (Simple Text Extraction)

```python
def parse_docx(self, file_path: str, use_markdown: bool = True) -> str:
    """Parse DOCX file and extract text content"""
    try:
        doc = Document(file_path)
        
        if use_markdown:
            # Simple text extraction (current behavior)
            text_content = ""
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content += " | ".join(row_text) + "\n"
        
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error parsing DOCX {file_path}: {e}")
        return ""
```

#### Structured Format (Layout Preservation)

```python
def _extract_structured_docx(self, doc) -> str:
    """Extract text from DOCX preserving structure and formatting"""
    text_parts = []
    
    # Extract paragraphs with formatting info
    for i, paragraph in enumerate(doc.paragraphs):
        if paragraph.text.strip():
            # Add paragraph structure info
            text_parts.append(f"[PARAGRAPH {i+1}] {paragraph.text}")
            
            # Add formatting information if available
            if paragraph.style:
                text_parts.append(f"[STYLE: {paragraph.style.name}]")
    
    # Extract tables with structure preservation
    for table_idx, table in enumerate(doc.tables):
        text_parts.append(f"\n=== TABLE {table_idx + 1} ===")
        
        for row_idx, row in enumerate(table.rows):
            row_data = []
            for cell_idx, cell in enumerate(row.cells):
                if cell.text.strip():
                    row_data.append(f"[C{cell_idx+1}] {cell.text.strip()}")
            
            if row_data:
                text_parts.append(f"[ROW {row_idx+1}] {' | '.join(row_data)}")
    
    return "\n".join(text_parts)
```

### Return Format

```python
def parse_document(self, file_path: str, use_markdown: bool = True) -> Dict[str, Any]:
    """Parse a document and return its content with metadata"""
    # ... parsing logic ...
    
    format_type = "markdown" if use_markdown else "structured"
    logger.info(f"Parsing document: {file_name} (format: {format_type})")
    
    return {
        'file_path': file_path,
        'file_name': file_name,
        'file_extension': file_extension,
        'text_content': text_content,
        'content_length': len(text_content),
        'word_count': len(text_content.split()) if text_content else 0,
        'parsing_method': 'fast',
        'format_used': format_type
    }
```

## Docling Parsing Implementation

### Core Classes

#### DoclingParser Class

```python
class DoclingParser:
    def __init__(self):
        self.summaries = summaries.copy()
        
        # Detect CUDA availability
        self.device = self._detect_device()
        print(f"Using device: {self.device}")
        
        # Initialize pipeline options with detected device
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            generate_picture_images=False,
            generate_page_images=False,
            do_formula_enrichment=True,
            table_structure_options={"do_cell_matching": True},
            accelerator_options=AcceleratorOptions(
                num_threads=4, 
                device=self.device
            ),
        )

        # Initialize format options for supported file types
        self.format_options = self._initialize_format_options()
        self.converter = DocumentConverter(format_options=self.format_options)
        
        # Supported file extensions
        self.supported_extensions = self._get_supported_extensions()
        print(f"Supported file formats: {', '.join(self.supported_extensions)}")
```

### CUDA Detection

```python
def _detect_device(self):
    """Detect the best available device (CUDA or CPU)"""
    if HAS_TORCH:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA available with {device_count} device(s)")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"  Device {i}: {device_name}")
            return AcceleratorDevice.CUDA
        else:
            print("CUDA not available, using CPU")
    else:
        print("PyTorch not available, using CPU")
    
    return AcceleratorDevice.CPU
```

### Format Options Initialization

```python
def _initialize_format_options(self):
    """Initialize format options for all supported file types"""
    format_options = {}
    
    # PDF format (always supported)
    format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=self.pipeline_options)
    
    # Additional formats if available
    if HAS_ADDITIONAL_FORMATS:
        try:
            # DOCX format
            format_options[InputFormat.DOCX] = DocxFormatOption()
            print("DOCX format support enabled")
        except Exception as e:
            print(f"Warning: Could not initialize DOCX format: {e}")
        
        try:
            # TXT format
            format_options[InputFormat.TXT] = TxtFormatOption()
            print("TXT format support enabled")
        except Exception as e:
            print(f"Warning: Could not initialize TXT format: {e}")
    else:
        print("Additional format options not available")
    
    return format_options
```

### Document Conversion

#### Main Conversion Method

```python
def parse_document(self, file_path: str, chunk_size: int = 3000, chunk_overlap: int = 200, use_markdown: bool = True) -> dict:
    """
    Convert any supported document to the same format as DocumentParser.
    Returns a dictionary with document content and metadata (no chunking).
    """
    import psutil
    
    if not self.is_supported_file(file_path):
        raise ValueError(f"Unsupported file format: {self.get_file_extension(file_path)}")
    
    file_extension = self.get_file_extension(file_path)
    file_name = os.path.basename(file_path)
    document_name = os.path.splitext(file_name)[0]
    
    # Convert document using docling
    print(f"Parsing {file_extension} document...")
    result = self.converter.convert(file_path)
    doc = result.document
    print(f"Document parsing complete. Pages: {len(doc.pages) if hasattr(doc, 'pages') else 'unknown'}")
    
    if use_markdown:
        # Export document to markdown to get the full text content
        text_content = doc.export_to_markdown(image_mode="embedded")
        
        # Only replace base64 images for PDF files (other formats may not have images)
        if file_extension == '.pdf':
            text_content = self.replace_base64_images(text_content, self.summaries.copy())
        
        print(f"Using markdown format for extraction")
    else:
        # Use structured document format - extract text from document structure
        text_content = self._extract_structured_text(doc)
        print(f"Using structured document format for extraction")
    
    # Return in the same format as DocumentParser
    document_dict = {
        'file_path': file_path,
        'file_name': file_name,
        'file_extension': file_extension,
        'text_content': text_content,
        'content_length': len(text_content),
        'word_count': len(text_content.split()) if text_content else 0,
        'parsing_method': 'docling',
        'format_used': 'markdown' if use_markdown else 'structured'
    }
    
    print(f"Successfully parsed {document_name} ({file_extension}) with Docling")
    return document_dict
```

#### Structured Text Extraction

```python
def _extract_structured_text(self, doc) -> str:
    """
    Extract text content from Docling document structure preserving layout information.
    This method preserves more structural information than markdown export.
    """
    text_parts = []
    
    # Extract text from document structure
    if hasattr(doc, 'texts'):
        # If document has texts attribute, use it
        for text_obj in doc.texts:
            if hasattr(text_obj, 'text'):
                text_parts.append(text_obj.text)
            elif hasattr(text_obj, 'content'):
                text_parts.append(text_obj.content)
    
    # If no texts attribute, try to extract from pages
    elif hasattr(doc, 'pages'):
        for page in doc.pages:
            if hasattr(page, 'texts'):
                for text_obj in page.texts:
                    if hasattr(text_obj, 'text'):
                        text_parts.append(text_obj.text)
                    elif hasattr(text_obj, 'content'):
                        text_parts.append(text_obj.content)
    
    # If still no text found, try to get text from document directly
    if not text_parts:
        if hasattr(doc, 'text'):
            text_parts.append(doc.text)
        elif hasattr(doc, 'content'):
            text_parts.append(doc.content)
        else:
            # Fallback to markdown export
            print("Warning: Could not extract structured text, falling back to markdown")
            return doc.export_to_markdown(image_mode="embedded")
    
    # Join text parts with appropriate separators
    structured_text = "\n\n".join(text_parts)
    
    return structured_text
```

## UI Integration

### Format Selection Implementation

```python
# Fast Parsing Format Selection
if current_method == "fast":
    st.info("⚡ **Fast parsing** - Uses PyMuPDF/docx libraries for all file types")
    
    # Add format selection for Fast parsing
    if 'fast_format' not in st.session_state:
        st.session_state.fast_format = 'markdown'
    
    format_option = st.radio(
        "Fast parsing format:",
        options=["Markdown format", "Structured format"],
        index=0 if st.session_state.fast_format == 'markdown' else 1,
        help="Markdown: Clean text extraction. Structured: Preserves layout, positioning, and document structure.",
        key="fast_format_radio"
    )
    
    if format_option == "Markdown format":
        st.session_state.fast_format = 'markdown'
        st.caption("📝 **Markdown**: Clean text format - Use for letters, reports, articles, and general text extraction")
    else:
        st.session_state.fast_format = 'structured'
        st.caption("🏗️ **Structured**: Preserves layout - Use for forms, tables, multi-column documents, and structured data")

# Docling Parsing Format Selection
elif current_method == "docling" and DOCLING_AVAILABLE:
    st.info("🔬 **Docling parsing** - Advanced analysis with CUDA acceleration")
    
    # Add format selection for Docling
    if 'docling_format' not in st.session_state:
        st.session_state.docling_format = 'markdown'
    
    format_option = st.radio(
        "Docling format:",
        options=["Markdown format", "Structured format"],
        index=0 if st.session_state.docling_format == 'markdown' else 1,
        help="Markdown: Best for simple documents, text extraction, and consistent results. Structured: Best for complex layouts, tables, forms, and documents where positioning matters.",
        key="docling_format_radio"
    )
```

### Parsing Logic Integration

```python
if parsing_method == 'docling':
    # Check if DoclingParser supports this file type
    if document_parser.is_supported_file(file_path):
        # Get the selected format
        use_markdown = st.session_state.get('docling_format', 'markdown') == 'markdown'
        # DoclingParser now returns a dictionary in the same format as DocumentParser
        parsed_doc = document_parser.parse_document(file_path, use_markdown=use_markdown)
        parsed_documents.append(parsed_doc)
    else:
        # For unsupported files, fall back to fast parsing
        file_extension = os.path.splitext(file_path)[1].lower()
        st.warning(f"⚠️ Docling parsing doesn't support {file_extension} files. Using Fast parsing for {os.path.basename(file_path)}")
        # Use the same format as selected for fast parsing
        use_markdown = st.session_state.get('fast_format', 'markdown') == 'markdown'
        parsed_doc = DocumentParser().parse_document(file_path, use_markdown=use_markdown)
        parsed_documents.append(parsed_doc)
else:
    # Get the selected format for fast parsing
    use_markdown = st.session_state.get('fast_format', 'markdown') == 'markdown'
    # DocumentParser returns document dictionary
    parsed_doc = document_parser.parse_document(file_path, use_markdown=use_markdown)
    parsed_documents.append(parsed_doc)
```

## Format Comparison

### Markdown Format Output Examples

#### Fast Parsing - PDF Markdown
```
Document Title
This is the main content of the document.

Page 2
More content continues here.

Table data | Column 2 | Column 3
Row 1 data | Value 1 | Value 2
Row 2 data | Value 3 | Value 4
```

#### Docling Parsing - PDF Markdown
```
# Document Title

This is the main content of the document with **bold text** and *italic text*.

## Section Header

More content continues here with proper markdown formatting.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Row 1 data | Value 1 | Value 2 |
| Row 2 data | Value 3 | Value 4 |
```

### Structured Format Output Examples

#### Fast Parsing - PDF Structured
```
=== PAGE 1 ===
[x:100,y:200] Document Title
[x:100,y:250] This is the main content of the document.
[x:100,y:300] More content continues here.

=== PAGE 2 ===
[x:100,y:200] Page 2
[x:100,y:250] Additional content here.
```

#### Fast Parsing - DOCX Structured
```
[PARAGRAPH 1] Document Title
[STYLE: Heading 1]
[PARAGRAPH 2] This is the main content of the document.
[PARAGRAPH 3] More content continues here.

=== TABLE 1 ===
[ROW 1] [C1] Column 1 | [C2] Column 2 | [C3] Column 3
[ROW 2] [C1] Row 1 data | [C2] Value 1 | [C3] Value 2
[ROW 3] [C1] Row 2 data | [C2] Value 3 | [C3] Value 4
```

#### Docling Parsing - Structured
```
Document Title

This is the main content of the document with preserved structure.

Section Header

More content continues here with structural information preserved.

Table data with structure information
Column headers and row data preserved
```

## Performance Considerations

### Fast Parsing
- **Speed**: Very fast processing
- **Memory**: Low memory usage
- **Accuracy**: Good for simple documents
- **CUDA**: Not available

### Docling Parsing
- **Speed**: Slower processing (especially without GPU)
- **Memory**: Higher memory usage
- **Accuracy**: Excellent for complex documents
- **CUDA**: Automatic detection and usage when available

## Error Handling

### File Format Validation
```python
def is_supported_file(self, file_path: str) -> bool:
    """Check if file format is supported"""
    extension = self.get_file_extension(file_path)
    return extension in self.supported_extensions
```

### Graceful Fallbacks
- Docling parsing falls back to Fast parsing for unsupported file types
- Structured format falls back to Markdown format if extraction fails
- CUDA falls back to CPU if not available

## Configuration Management

### Session State Variables
- `parsing_method`: 'fast' or 'docling'
- `fast_format`: 'markdown' or 'structured'
- `docling_format`: 'markdown' or 'structured'

### Default Values
- Parsing method: 'fast'
- Format: 'markdown' for both methods

## Best Practices

### When to Use Markdown Format
- Simple documents (letters, reports, articles)
- General text extraction
- Consistent results across document types
- Basic data extraction (names, dates, descriptions)

### When to Use Structured Format
- Complex layouts and multi-column documents
- Tables and forms
- Documents where positioning matters
- Structured data extraction (table rows, form fields)

### When to Use Fast Parsing
- Simple documents
- Speed is critical
- Limited computational resources
- Standard document types (.pdf, .docx, .doc)

### When to Use Docling Parsing
- Complex documents with advanced layouts
- High accuracy requirements
- GPU acceleration available
- Advanced document types (.ppt, .pptx, .txt)

## Troubleshooting

### Common Issues
1. **CUDA not detected**: Check PyTorch installation and GPU drivers
2. **Docling import errors**: Ensure Docling library is properly installed
3. **Memory issues**: Use Fast parsing for large documents
4. **Format selection not working**: Check session state initialization

### Debug Information
- Console output shows device detection and format selection
- Logging provides detailed parsing information
- Error messages indicate specific failure points
