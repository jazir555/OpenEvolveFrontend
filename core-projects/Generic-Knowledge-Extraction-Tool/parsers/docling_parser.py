# Try to import docling components with fallback handling
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, AcceleratorDevice, AcceleratorOptions
    from docling.chunking import HierarchicalChunker
    DOCLING_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Docling core components not available: {e}")
    DOCLING_IMPORT_SUCCESS = False
    # Create dummy classes for fallback
    class DocumentConverter:
        def __init__(self, *args, **kwargs): pass
        def convert(self, *args, **kwargs): return None
    class PdfFormatOption:
        def __init__(self, *args, **kwargs): pass
    class InputFormat:
        PDF = "pdf"
        DOCX = "docx"
        TXT = "txt"
    class PdfPipelineOptions:
        def __init__(self, *args, **kwargs): pass
    class AcceleratorDevice:
        CPU = "cpu"
        CUDA = "cuda"
    class AcceleratorOptions:
        def __init__(self, *args, **kwargs): pass
    class HierarchicalChunker:
        def __init__(self, *args, **kwargs): pass

from langchain.schema import Document
import re
import os
from collections import OrderedDict

# Try to import additional format options
try:
    from docling.document_converter import DocxFormatOption, TxtFormatOption
    HAS_ADDITIONAL_FORMATS = True
except ImportError:
    HAS_ADDITIONAL_FORMATS = False
    print("Warning: Additional format options not available. Only PDF will be supported.")

# Try to import basic document converter for fallback
try:
    from docling.document_converter import DocumentConverter
    HAS_BASIC_CONVERTER = True
except ImportError:
    HAS_BASIC_CONVERTER = False
    print("Warning: Basic document converter not available.")

# Try to import torch for CUDA detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. CUDA detection disabled.")

try:
    from src.summaries_images import summaries
    summaries = OrderedDict(summaries)
except ImportError:
    summaries = OrderedDict()


class DoclingParser:
    def __init__(self):
        self.summaries = summaries.copy()
        self.docling_available = DOCLING_IMPORT_SUCCESS
        
        if not self.docling_available:
            # If docling is not available, provide minimal functionality
            self.supported_extensions = ['.pdf']  # Only PDF supported without full docling
            self.converter = None
            return
        
        # Detect CUDA availability
        self.device = self._detect_device()
        
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
    
# **ACTUAL INTEGRATION**: Adaptive MDAP for Docling Parser
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class DoclingParser:
    """Parser for extracting content from various document formats using Docling."""
    
    def __init__(self):
        """Initialize the Docling parser with pipeline options."""
        self.device = self._detect_device()
        self.pipeline_options = self._create_pipeline_options()
        # Initialize format options for supported file types
        self.format_options = self._initialize_format_options()
        self.converter = DocumentConverter(format_options=self.format_options)
        
        # Supported file extensions
        self.supported_extensions = self._get_supported_extensions()
    
    def _detect_device(self):
        """Detect the best available device (CUDA or CPU)"""
        if HAS_TORCH:
            if torch.cuda.is_available():
                return AcceleratorDevice.CUDA
            else:
                return AcceleratorDevice.CPU
        else:
            return AcceleratorDevice.CPU
    
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
            except Exception as e:
                pass
            
            try:
                # TXT format
                format_options[InputFormat.TXT] = TxtFormatOption()
            except Exception as e:
                pass
        
        # If additional formats aren't available but basic converter is, 
        # docling will use default format options for .docx and .txt
        return format_options
    
    def _get_supported_extensions(self):
        """Get list of supported file extensions"""
        extensions = ['.pdf']  # PDF is always supported
        
        # Always support .docx and .txt if basic converter is available
        if HAS_BASIC_CONVERTER and self.docling_available:
            extensions.extend(['.docx', '.txt'])
        
        # Add additional formats if available
        if HAS_ADDITIONAL_FORMATS and self.docling_available:
            # Check if DOC format is supported (usually through DOCX)
            try:
                # Try to add DOC support (often handled by DOCX converter)
                extensions.append('.doc')
            except:
                pass
            
            # Check if PPT format is supported
            try:
                extensions.append('.ppt')
                extensions.append('.pptx')
            except:
                pass
        
        return extensions

    def replace_base64_images(self, md_text, summary_dict):
        pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

        def replacement(match):
            if summary_dict:
                key, value = summary_dict.popitem(last=False)
                return f"\n\n{value}\n\n"
            else:
                return "\n\n[Image removed - no summary available]\n\n"

        return re.sub(pattern, replacement, md_text)

    def get_file_extension(self, file_path: str) -> str:
        """Get file extension from file path"""
        return os.path.splitext(file_path)[1].lower()
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file format is supported"""
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions
    
    def convert_document_to_markdown(self, file_path: str, output_path: str = None) -> str:
        """Convert any supported document to markdown"""
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file format: {self.get_file_extension(file_path)}")
        
        if output_path is None:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(os.path.dirname(file_path), f"{file_name}.md")
        
        result = self.converter.convert(file_path)
        markdown_text = result.document.export_to_markdown(image_mode="embedded")
        
        # Only replace base64 images for PDF files (other formats may not have images)
        if self.get_file_extension(file_path) == '.pdf':
            new_markdown = self.replace_base64_images(markdown_text, self.summaries.copy())
            markdown_text = new_markdown
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        return markdown_text

    def parse_document(self, file_path: str, chunk_size: int = 3000, chunk_overlap: int = 200, use_markdown: bool = True) -> dict:
        """
        Convert any supported document to the same format as DocumentParser.
        Returns a dictionary with document content and metadata (no chunking).
        
        Args:
            file_path: Path to the document file
            chunk_size: Not used (kept for compatibility)
            chunk_overlap: Not used (kept for compatibility)  
            use_markdown: If True, uses markdown format. If False, uses structured document format.
        """
        import psutil
        
        if not self.docling_available:
            raise ValueError("Docling is not available. Please install required dependencies or use Fast parsing.")
        
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file format: {self.get_file_extension(file_path)}")
        
        file_extension = self.get_file_extension(file_path)
        file_name = os.path.basename(file_path)
        document_name = os.path.splitext(file_name)[0]
        
        # Check container memory limits and available memory
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        
        # Check if running in container with cgroup limits (v1 and v2)
        try:
            # Try cgroup v1 first
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                cgroup_limit = int(f.read().strip())
        except (FileNotFoundError, PermissionError):
            try:
                # Try cgroup v2 
                with open('/sys/fs/cgroup/memory.max', 'r') as f:
                    cgroup_limit = f.read().strip()
                    if cgroup_limit != 'max':
                        cgroup_limit = int(cgroup_limit)
            except (FileNotFoundError, PermissionError):
                pass
        
        try:
            # Try cgroup v1 usage
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                cgroup_usage = int(f.read().strip())
        except (FileNotFoundError, PermissionError):
            try:
                # Try cgroup v2 usage 
                with open('/sys/fs/cgroup/memory.current', 'r') as f:
                    cgroup_usage = int(f.read().strip())
            except (FileNotFoundError, PermissionError):
                pass
        
        # Convert document using docling
        result = self.converter.convert(file_path)
        doc = result.document
        
        if use_markdown:
            # Export document to markdown to get the full text content
            text_content = doc.export_to_markdown(image_mode="embedded")
            
            # Only replace base64 images for PDF files (other formats may not have images)
            if file_extension == '.pdf':
                text_content = self.replace_base64_images(text_content, self.summaries.copy())
        else:
            # Use structured document format - extract text from document structure
            text_content = self._extract_structured_text(doc)
        
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
        
        return document_dict
    
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
                return doc.export_to_markdown(image_mode="embedded")
        
        # Join text parts with appropriate separators
        structured_text = "\n\n".join(text_parts)
        
        return structured_text