# knowledge_engine/document_loader.py

import os
import re
import aiohttp
import aiofiles
import shutil
import logging
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from urllib.parse import urlparse, unquote
from datetime import datetime

# Dependencies that need to be installed:
# PyPDF2, reportlab, docling, aiohttp, aiofiles

try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 is not installed. PDF metadata reading will not work.")
    PyPDF2 = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("Warning: reportlab is not installed. Text to PDF conversion will not work.")
    REPORTLAB_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling package not available. Document conversion will be disabled.")


def read_pdf_metadata(file_path: Path) -> dict:
    """Read PDF metadata with proper encoding handling."""
    if not PyPDF2:
        return {"title": "PyPDF2 not installed", "authors": [], "year": "", "first_lines": []}
    try:
        print(f"\nAttempting to read PDF metadata from: {file_path}")
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            info = pdf_reader.metadata
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()
            lines = text.split("\n")[:10]

            title = None
            authors = []

            if info:
                title = info.get("/Title", "").strip().replace("\x00", "")
                author = info.get("/Author", "").strip().replace("\x00", "")
                if author:
                    authors = [author]

            if not title and lines:
                title = lines[0].strip()

            if not authors and len(lines) > 1:
                for line in lines[1:3]:
                    if "author" in line.lower() or "by" in line.lower():
                        authors = [line.strip()]
                        break

            return {
                "title": title if title else "Unknown Title",
                "authors": authors if authors else ["Unknown Author"],
                "year": info.get("/CreationDate", "")[:4] if info else "Unknown Year",
                "first_lines": lines,
            }

    except Exception as e:
        print(f"\nError reading PDF: {str(e)}")
        return {
            "title": "Error reading PDF",
            "authors": ["Unknown"],
            "year": "Unknown",
            "first_lines": [],
        }


class PDFConverter:
    """
    PDF conversion utility class.

    Provides methods to convert Office documents and text files to PDF format.
    """

    OFFICE_FORMATS = { ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    TEXT_FORMATS = { ".txt", ".md"}
    logger = logging.getLogger(__name__)

    def convert_to_pdf(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
    ) -> Path:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        ext = file_path.suffix.lower()

        if ext in self.OFFICE_FORMATS:
            return self.convert_office_to_pdf(file_path, output_dir)
        elif ext in self.TEXT_FORMATS:
            return self.convert_text_to_pdf(file_path, output_dir)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(self.OFFICE_FORMATS | self.TEXT_FORMATS)}"
            )

    def convert_office_to_pdf(
        self, doc_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        try:
            # Convert to Path object for easier handling
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "pdf_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Check if LibreOffice is available
            libreoffice_available = False
            working_libreoffice_cmd: Optional[str] = None

            # Prepare subprocess parameters to hide console window on Windows
            subprocess_kwargs: Dict[str, Any] = {
                "capture_output": True,
                "check": True,
                "timeout": 10,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = (
                    0x08000000  # subprocess.CREATE_NO_WINDOW
                )

            try:
                result = subprocess.run(
                    ["libreoffice", "--version"], **subprocess_kwargs
                )
                libreoffice_available = True
                working_libreoffice_cmd = "libreoffice"
                self.logger.info(f"LibreOffice detected: {result.stdout.strip()}")
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                self.logger.debug("LibreOffice not available using default command.")

            # Try alternative commands for LibreOffice
            if not libreoffice_available:
                for cmd in ["soffice", "libreoffice"]:
                    try:
                        result = subprocess.run([cmd, "--version"], **subprocess_kwargs)
                        libreoffice_available = True
                        working_libreoffice_cmd = cmd
                        self.logger.info(
                            f"LibreOffice detected with command '{cmd}': {result.stdout.strip()}"
                        )
                        break
                    except (
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                        subprocess.TimeoutExpired,
                    ):
                        continue

            if not libreoffice_available:
                raise RuntimeError(
                    "LibreOffice is required for Office document conversion but was not found."
                )

            # Create temporary directory for PDF conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Convert to PDF using LibreOffice
                self.logger.info(f"Converting {doc_path.name} to PDF using LibreOffice...")

                commands_to_try = [working_libreoffice_cmd]
                if working_libreoffice_cmd == "libreoffice":
                    commands_to_try.append("soffice")
                else:
                    commands_to_try.append("libreoffice")

                conversion_successful = False
                for cmd in commands_to_try:
                    if cmd is None:
                        continue
                    try:
                        convert_cmd = [
                            cmd,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(doc_path),
                        ]

                        convert_subprocess_kwargs: Dict[str, Any] = {
                            "capture_output": True,
                            "text": True,
                            "timeout": 60,
                            "encoding": "utf-8",
                            "errors": "ignore",
                        }

                        if platform.system() == "Windows":
                            convert_subprocess_kwargs["creationflags"] = (
                                0x08000000
                            )

                        result = subprocess.run(
                            convert_cmd, **convert_subprocess_kwargs
                        )

                        if result.returncode == 0:
                            conversion_successful = True
                            self.logger.info(
                                f"Successfully converted {doc_path.name} to PDF"
                            )
                            break
                        else:
                            self.logger.warning(
                                f"LibreOffice command '{cmd}' failed: {result.stderr}"
                            )
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"LibreOffice command '{cmd}' timed out")
                    except Exception as e:
                        self.logger.error(
                            f"LibreOffice command '{cmd}' failed with exception: {e}"
                        )

                if not conversion_successful:
                    raise RuntimeError(
                        f"LibreOffice conversion failed for {doc_path.name}."
                    )

                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated."
                    )

                pdf_path = pdf_files[0]
                self.logger.info(
                    f"Generated PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)"
                )

                if pdf_path.stat().st_size < 100:
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted."
                    )

                final_pdf_path = base_output_dir / f"{name_without_suff}.pdf"
                shutil.copy2(pdf_path, final_pdf_path)

                return final_pdf_path

        except Exception as e:
            self.logger.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise

    def convert_text_to_pdf(
        self, text_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab is required for text-to-PDF conversion.")
        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            supported_text_formats = {".txt", ".md"}
            if text_path.suffix.lower() not in supported_text_formats:
                raise ValueError(f"Unsupported text format: {text_path.suffix}")

            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                for encoding in ["gbk", "latin-1", "cp1252"]:
                    try:
                        with open(text_path, "r", encoding=encoding) as f:
                            text_content = f.read()
                        self.logger.info(f"Successfully read file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        f"Could not decode text file {text_path.name} with any supported encoding"
                    )

            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = text_path.parent / "pdf_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = base_output_dir / f"{text_path.stem}.pdf"

            self.logger.info(f"Converting {text_path.name} to PDF...")

            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                leftMargin=inch,
                rightMargin=inch,
                topMargin=inch,
                bottomMargin=inch,
            )

            styles = getSampleStyleSheet()
            normal_style = styles["Normal"]
            heading_style = styles["Heading1"]

            try:
                system = platform.system()
                if system == "Windows":
                    for font_name in ["SimSun", "SimHei", "Microsoft YaHei"]:
                        try:
                            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                            normal_style.fontName = font_name
                            heading_style.fontName = font_name
                            break
                        except (ImportError, RuntimeError, KeyError):
                            continue
                elif system == "Darwin":
                    for font_name in ["STSong-Light", "STHeiti"]:
                        try:
                            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                            normal_style.fontName = font_name
                            heading_style.fontName = font_name
                            break
                        except (ImportError, RuntimeError, KeyError):
                            continue
            except (ImportError, RuntimeError, KeyError):
                self.logger.debug("Font registration failed; using default fonts.")

            story = []

            if text_path.suffix.lower() == ".md":
                lines = text_content.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 12))
                        continue

                    if line.startswith("#"):
                        level = len(line) - len(line.lstrip("#"))
                        header_text = line.lstrip("#").strip()
                        if header_text:
                            header_style = ParagraphStyle(
                                name=f"Heading{level}",
                                parent=heading_style,
                                fontSize=max(16 - level, 10),
                                spaceAfter=8,
                                spaceBefore=16 if level <= 2 else 12,
                            )
                            story.append(Paragraph(header_text, header_style))
                    else:
                        processed_line = self._process_inline_markdown(line)
                        story.append(Paragraph(processed_line, normal_style))
                        story.append(Spacer(1, 6))
            else:
                self.logger.info(
                    f"Processing plain text file with {len(text_content)} characters..."
                )
                lines = text_content.split("\n")
                line_count = 0
                for line in lines:
                    line = line.rstrip()
                    line_count += 1
                    if not line.strip():
                        story.append(Spacer(1, 6))
                        continue
                    safe_line = (
                        line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    story.append(Paragraph(safe_line, normal_style))
                    story.append(Spacer(1, 3))
                self.logger.info(f"Added {line_count} lines to PDF")
                if not story:
                    story.append(Paragraph("(Empty text file)", normal_style))

            doc.build(story)
            self.logger.info(
                f"Successfully converted {text_path.name} to PDF ({pdf_path.stat().st_size / 1024:.1f} KB)"
            )

            if not pdf_path.exists() or pdf_path.stat().st_size < 100:
                raise RuntimeError(
                    f"PDF conversion failed for {text_path.name} - generated PDF is empty or corrupted."
                )

            return pdf_path

        except Exception as e:
            self.logger.error(f"Error in convert_text_to_pdf: {str(e)}")
            raise

    @staticmethod
    def _process_inline_markdown(text: str) -> str:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'
        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)
        return text



class URLExtractor:
    """URL Extractor"""

    URL_PATTERNS = [
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/(?:[-\w._~!$&'()*+,;=:@]|%[\da-fA-F]{2})*)*(?:\?(?:[-\w._~!$&'()*+,;=:@/?]|%[\da-fA-F]{2})*)?(?:#(?:[-\w._~!$&'()*+,;=:@/?]|%[\da-fA-F]{2})*)?",
        r"ftp://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/(?:[-\w._~!$&'()*+,;=:@]|%[\da-fA-F]{2})*)*",
        r"(?<!\S)(?:www\.)?[-\w]+(?:\.[-\w]+)+/(?:[-\w._~!$&'()*+,;=:@/]|%[\da-fA-F]{2})+",
    ]

    @staticmethod
    def convert_arxiv_url(url: str) -> str:
        """Converts arXiv abstract URLs to PDF download links."""
        arxiv_pattern = r"arxiv\.org/abs/(\d+\.\d+)(?:v\d+)?"
        match = re.search(arxiv_pattern, url, re.IGNORECASE)
        if match:
            paper_id = match.group(1)
            return f"https://arxiv.org/pdf/{paper_id}.pdf"
        return url

    @classmethod
    def extract_urls(cls, text: str) -> List[str]:
        """Extracts URLs from text."""
        urls = []
        at_url_pattern = r"@(https?://[^\s]+)"
        at_matches = re.findall(at_url_pattern, text, re.IGNORECASE)
        for match in at_matches:
            url = cls.convert_arxiv_url(match.rstrip("/"))
            urls.append(url)

        for pattern in cls.URL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if not match.startswith(("http://", "https://", "ftp://")):
                    if match.startswith("www."):
                        match = "https://" + match
                    else:
                        match = "https://" + match
                url = cls.convert_arxiv_url(match.rstrip("/"))
                urls.append(url)

        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        return unique_urls

    @staticmethod
    def infer_filename_from_url(url: str) -> str:
        """Infers a filename from a URL."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = os.path.basename(path)
        if "arxiv.org" in parsed.netloc and "/pdf/" in path:
            if filename:
                if not filename.lower().endswith((".pdf", ".doc", ".docx", ".txt")):
                    filename = f"{filename}.pdf"
            else:
                path_parts = [p for p in path.split("/") if p]
                if path_parts and path_parts[-1]:
                    filename = f"{path_parts[-1]}.pdf"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"arxiv_paper_{timestamp}.pdf"
        elif not filename or "." not in filename:
            domain = parsed.netloc.replace("www.", "").replace(".", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not path or path == "/":
                filename = f"{domain}_{timestamp}.html"
            else:
                path_parts = [p for p in path.split("/") if p]
                if path_parts:
                    filename = f"{path_parts[-1]}_{timestamp}"
                else:
                    filename = f"{domain}_{timestamp}"
                if "." not in filename:
                    if "/pdf/" in path.lower() or path.lower().endswith("pdf"):
                        filename += ".pdf"
                    elif any(
                        ext in path.lower() for ext in ["/doc/", "/word/", ".docx"]
                    ):
                        filename += ".docx"
                    elif any(
                        ext in path.lower()
                        for ext in ["/ppt/", "/powerpoint/", ".pptx"]
                    ):
                        filename += ".pptx"
                    elif any(ext in path.lower() for ext in ["/csv/", ".csv"]):
                        filename += ".csv"
                    elif any(ext in path.lower() for ext in ["/zip/", ".zip"]):
                        filename += ".zip"
                    else:
                        filename += ".html"
        return filename


class SimplePdfConverter:
    """A simple PDF to Markdown converter using PyPDF2."""

    def convert_pdf_to_markdown(
        self, input_file: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        if not PyPDF2:
            return {"success": False, "error": "PyPDF2 package is not available"}

        try:
            if not os.path.exists(input_file):
                return {
                    "success": False,
                    "error": f"Input file not found: {input_file}",
                }
            if not output_file:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.md"
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            start_time = datetime.now()
            with open(input_file, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"## Page {page_num}\n\n{text.strip()}\n\n")
            markdown_content = f"# Extracted from {os.path.basename(input_file)}\n\n"
            markdown_content += f"*Total pages: {len(pdf_reader.pages)}*\n\n"
            markdown_content += "---\n\n"
            markdown_content += "".join(text_content)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            duration = (datetime.now() - start_time).total_seconds()
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            return {
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "input_size": input_size,
                "output_size": output_size,
                "duration": duration,
                "markdown_content": markdown_content,
                "pages_extracted": len(pdf_reader.pages),
            }
        except Exception as e:
            return {
                "success": False,
                "input_file": input_file,
                "error": f"Conversion failed: {str(e)}",
            }


class DoclingConverter:
    """A document to Markdown converter using the docling library."""

    def __init__(self):
        if not DOCLING_AVAILABLE:
            raise ImportError("docling package is not available. Please install it first.")
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = False
        pdf_pipeline_options.do_table_structure = False
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_pipeline_options
                    )
                }
            )
        except (ImportError, RuntimeError):
            self.converter = DocumentConverter()

    def convert_to_markdown(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        extract_images: bool = True,
    ) -> Dict[str, Any]:
        if not DOCLING_AVAILABLE:
            return {"success": False, "error": "docling package is not available"}
        try:
            if not urlparse(input_file).scheme in ("http", "https"):
                if not os.path.exists(input_file):
                    return {
                        "success": False,
                        "error": f"Input file not found: {input_file}",
                    }
            if not output_file:
                if urlparse(input_file).scheme in ("http", "https"):
                    filename = URLExtractor.infer_filename_from_url(input_file)
                    base_name = os.path.splitext(filename)[0]
                else:
                    base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.md"
            output_dir = os.path.dirname(output_file) or "."
            os.makedirs(output_dir, exist_ok=True)
            start_time = datetime.now()
            result = self.converter.convert(input_file)
            doc = result.document
            markdown_content = doc.export_to_markdown()
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            duration = (datetime.now() - start_time).total_seconds()
            if urlparse(input_file).scheme in ("http", "https"):
                input_size = 0
            else:
                input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            return {
                "success": True,
                "input_file": input_file,
                "output_file": output_file,
                "input_size": input_size,
                "output_size": output_size,
                "duration": duration,
                "markdown_content": markdown_content,
            }
        except Exception as e:
            return {
                "success": False,
                "input_file": input_file,
                "error": f"Conversion failed: {str(e)}",
            }


async def download_file(url: str, destination: str) -> Dict[str, Any]:
    """Downloads a file from a URL."""
    start_time = datetime.now()
    chunk_size = 8192
    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content_type = response.headers.get(
                    "Content-Type", "application/octet-stream"
                )
                parent_dir = os.path.dirname(destination)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                downloaded = 0
                async with aiofiles.open(destination, "wb") as file:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await file.write(chunk)
                        downloaded += len(chunk)
                duration = (datetime.now() - start_time).total_seconds()
                return {
                    "success": True,
                    "url": url,
                    "destination": destination,
                    "size": downloaded,
                    "content_type": content_type,
                    "duration": duration,
                    "speed": downloaded / duration if duration > 0 else 0,
                }
    except aiohttp.ClientError as e:
        return {
            "success": False,
            "url": url,
            "destination": destination,
            "error": f"Network error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "destination": destination,
            "error": f"Download error: {str(e)}",
        }
