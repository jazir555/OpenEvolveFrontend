#!/usr/bin/env python3
"""
Unified Knowledge Extraction Agent UI
Combined configuration and extraction interface.
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Ui App
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import streamlit as st
import json
import os
import glob
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import time
import logging

# Import our modules
from core.model_generator import ModelGenerator
from parsers.document_parser import DocumentParser
from utils.messaging_system import get_messenger
try:
    # Try to import docling with comprehensive error handling
    import os
    import sys
    
    # Temporarily suppress warnings and errors during import
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    sys.stderr = open(os.devnull, 'w')
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from parsers.docling_parser import DoclingParser
        # Test if DoclingParser is actually functional
        test_parser = DoclingParser()
        DOCLING_AVAILABLE = test_parser.docling_available
        if not DOCLING_AVAILABLE:
            print("Warning: DoclingParser imported but not functional (missing dependencies)")
    except Exception as e:
        DoclingParser = None
        DOCLING_AVAILABLE = False
        print(f"Warning: DoclingParser not available: {e}")
    finally:
        # Restore stdout and stderr
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        
except Exception as e:
    print(f"Warning: Could not import DoclingParser: {e}")
    DoclingParser = None
    DOCLING_AVAILABLE = False

from ai.extractors.openai_extractor import OpenAIExtractor
from ai.extractors.claude_extractor import ClaudeExtractor
from core.text_description_parser import TextDescriptionParser
from utils.prompts.text_description_prompts import get_example_descriptions, get_parsing_tips

# Import Case 1 multi-type document extraction
try:
    from extraction.case1_classifier import Case1Extractor
    CASE1_AVAILABLE = True
except ImportError as e:
    Case1Extractor = None
    CASE1_AVAILABLE = False
    print(f"Warning: Case1Extractor not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_config():
    """Get API configuration based on whether Azure endpoint is selected"""
    import os
    use_azure = st.session_state.get('use_azure', False)
    
    if use_azure:
        # Try secrets first, then fall back to environment variables
        try:
            azure_api_key = st.secrets["AZURE_API_KEY"]
        except:
            azure_api_key = os.getenv('AZURE_API_KEY')
            
        return {
            'use_azure': True,
            'api_key': azure_api_key,
            'azure_endpoint': "https://haagahelia-poc-gaik.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?",
            'azure_audio_endpoint': "https://haagahelia-poc-gaik.openai.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01",
            'api_version': "2024-12-01-preview",
            'model': 'gpt-4.1',  # Chat completion model
        }
    else:
        # Try secrets first, then fall back to environment variables
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            
        return {
            'use_azure': False,
            'api_key': openai_api_key,
            'model': 'gpt-4.1-2025-04-14',  # Chat completion model
        }

# Page configuration
st.set_page_config(
    page_title="Knowledge Extraction Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, appealing design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.3rem;
        color: #2e7d32;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2e7d32;
        padding-left: 1rem;
        font-weight: bold;
    }
    .field-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .compact-input {
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .results-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #28a745;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
    }
    .results-header {
        font-size: 1.5rem;
        color: #28a745;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 2px solid #28a745;
        padding-bottom: 0.5rem;
    }
    .result-summary {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .download-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #2196f3;
    }
    .certificate-green {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #155724;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.2);
    }
    .certificate-red {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #721c24;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(220, 53, 69, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Configuration"
    if 'fields' not in st.session_state:
        st.session_state.fields = []
    if 'use_case' not in st.session_state:
        st.session_state.use_case = ""
    if 'description' not in st.session_state:
        st.session_state.description = ""
    if 'main_model_name' not in st.session_state:
        st.session_state.main_model_name = ""
    if 'additional_instructions' not in st.session_state:
        st.session_state.additional_instructions = ""
    if 'extraction_purpose' not in st.session_state:
        st.session_state.extraction_purpose = ""
    if 'document_type' not in st.session_state:
        st.session_state.document_type = ""
    if 'custom_instructions' not in st.session_state:
        st.session_state.custom_instructions = ""
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'rebuild_models' not in st.session_state:
        st.session_state.rebuild_models = False
    if 'model_generation_model' not in st.session_state:
        st.session_state.model_generation_model = 'claude-sonnet-4-20250514'
    if 'extraction_model' not in st.session_state:
        st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
    if 'use_azure' not in st.session_state:
        st.session_state.use_azure = True
    
    # Text description mode variables
    if 'configuration_mode' not in st.session_state:
        st.session_state.configuration_mode = "fields"  # "fields" or "text_description"
    if 'text_description' not in st.session_state:
        st.session_state.text_description = ""
    if 'parsed_fields' not in st.session_state:
        st.session_state.parsed_fields = []
    if 'description_parsing_model' not in st.session_state:
        st.session_state.description_parsing_model = 'claude-sonnet-4-20250514'
    if 'parsing_in_progress' not in st.session_state:
        st.session_state.parsing_in_progress = False
    
    # Document parsing method
    if 'parsing_method' not in st.session_state:
        st.session_state.parsing_method = "fast"  # "fast" or "docling"
    
    # Multi-type document extraction
    if 'extraction_type' not in st.session_state:
        st.session_state.extraction_type = "single_type"  # "single_type", "multi_type", or "multi_type_with_relationships"

def ensure_use_cases_folder():
    """Ensure templates folder exists at application start"""
    use_cases_dir = "templates"
    if not os.path.exists(use_cases_dir):
        os.makedirs(use_cases_dir)
        st.success(f"✅ Created {use_cases_dir} folder")
    return use_cases_dir

def parse_text_description():
    """Parse text description and extract field configuration"""
    try:
        if not st.session_state.text_description.strip():
            st.error("Please enter a description first")
            return False
        
        # Show parsing progress
        st.session_state.parsing_in_progress = True
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Parsing description...")
        progress_bar.progress(20)
        
        # Initialize parser
        api_config = get_api_config()
        parser = TextDescriptionParser(
            model_selection=st.session_state.description_parsing_model,
            api_config=api_config
        )
        
        status_text.text("🔍 Analyzing description with AI...")
        progress_bar.progress(50)
        
        # Parse description
        config = parser.parse_extraction_description(
            description=st.session_state.text_description,
            use_case=st.session_state.use_case,
            context=st.session_state.description
        )
        
        status_text.text("🔍 Processing parsed fields...")
        progress_bar.progress(80)
        
        # Extract parsed fields
        parsed_fields = config['extraction_config']['parsed_fields']
        st.session_state.parsed_fields = parsed_fields
        
        # Update fields for backward compatibility
        st.session_state.fields = parsed_fields
        
        # Update model name if not set
        if not st.session_state.main_model_name:
            st.session_state.main_model_name = config['extraction_config']['main_model_name']
        
        status_text.text("✅ Description parsed successfully!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.parsing_in_progress = False
        return True
        
    except Exception as e:
        st.session_state.parsing_in_progress = False
        st.error(f"Error parsing description: {str(e)}")
        return False

def create_use_case_folder(use_case_name: str) -> str:
    """Create folder for specific use case"""
    use_cases_dir = ensure_use_cases_folder()
    # Create safe folder name
    safe_name = use_case_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    
    use_case_folder = os.path.join(use_cases_dir, safe_name)
    if not os.path.exists(use_case_folder):
        os.makedirs(use_case_folder)
    
    return use_case_folder

def get_use_case_path(use_case_name: str, filename: str) -> str:
    """Get full path for a file in the use case folder"""
    use_case_folder = create_use_case_folder(use_case_name)
    return os.path.join(use_case_folder, filename)

def load_extraction_context_from_current_config():
    """Load extraction context from the current configuration for the extraction phase"""
    if not st.session_state.use_case:
        return  # No use case defined yet
    
    try:
        # Get the config file path for current use case
        config_path = get_use_case_path(st.session_state.use_case, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                extraction_config = config_data.get('extraction_config', {})
                
                # Check if this is a multi-type extraction use case
                extraction_type = extraction_config.get('extraction_type', 'single_type')
                
                # IMPORTANT: Override session state with the extraction type from the config
                # This ensures we load the use case with the correct extraction type
                logger.info(f"Loading use case '{st.session_state.use_case}' - Config extraction_type: {extraction_type}, Session extraction_type: {st.session_state.extraction_type}")
                st.session_state.extraction_type = extraction_type
                logger.info(f"After override - Session extraction_type: {st.session_state.extraction_type}")
                
                if extraction_type == 'multi_type':
                    # For multi-type extraction, load the main config but don't try to load models
                    # The models will be loaded dynamically during extraction
                    st.session_state.extraction_purpose = extraction_config.get('purpose_of_extraction', '')
                    
                    # Handle document_type - remove the appended warning if present
                    doc_type_raw = extraction_config.get('document_type', '')
                    if ". Do not attempt to extract data from non-related documents" in doc_type_raw:
                        st.session_state.document_type = doc_type_raw.split(". Do not attempt to extract data from non-related documents")[0]
                    else:
                        st.session_state.document_type = doc_type_raw
                    
                    st.session_state.custom_instructions = extraction_config.get('additional_instructions', '')
                    
                    # Load text description for multi-type
                    st.session_state.text_description = extraction_config.get('text_description', '')
                
                elif extraction_type == 'multi_type_with_relationships':
                    # For Case 2 hierarchical extraction, load the main config
                    st.session_state.extraction_purpose = extraction_config.get('purpose_of_extraction', '')
                    
                    # Handle document_type - remove the appended warning if present
                    doc_type_raw = extraction_config.get('document_type', '')
                    if ". Do not attempt to extract data from non-related documents" in doc_type_raw:
                        st.session_state.document_type = doc_type_raw.split(". Do not attempt to extract data from non-related documents")[0]
                    else:
                        st.session_state.document_type = doc_type_raw
                    
                    st.session_state.custom_instructions = extraction_config.get('additional_instructions', '')
                    
                    # Set extraction type for Case 2
                    st.session_state.extraction_type = 'multi_type_with_relationships'
                    
                    # Load text description for Case 2
                    st.session_state.text_description = extraction_config.get('text_description', '')
                    
                    # Mark as loaded use case
                    st.session_state.use_case_loaded = True
                    
                else:
                    # For single-type extraction, use the original logic
                    # Load the three extraction context fields
                    st.session_state.extraction_purpose = extraction_config.get('purpose_of_extraction', '')
                    
                    # Handle document_type - remove the appended warning if present
                    doc_type_raw = extraction_config.get('document_type', '')
                    if ". Do not attempt to extract data from non-related documents" in doc_type_raw:
                        st.session_state.document_type = doc_type_raw.split(". Do not attempt to extract data from non-related documents")[0]
                    else:
                        st.session_state.document_type = doc_type_raw
                    
                    st.session_state.custom_instructions = extraction_config.get('additional_instructions', '')
                    
    except Exception as e:
        # If there's any error loading, just use empty defaults
        st.session_state.extraction_purpose = ""
        st.session_state.document_type = ""
        st.session_state.custom_instructions = ""

def save_extraction_context_to_config():
    """Save the current extraction context (purpose, document type, custom instructions) to config.json"""
    if not st.session_state.use_case:
        return  # No use case defined yet
    
    try:
        # Get the config file path for current use case
        config_path = get_use_case_path(st.session_state.use_case, "config.json")
        if os.path.exists(config_path):
            # Load existing config
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            # Create a fresh config.json from the current session configuration
            config_data = export_configuration()
        
        # Update the extraction context fields
        extraction_config = config_data.get('extraction_config', {})
        
        # Get session state values with fallbacks
        purpose = getattr(st.session_state, 'extraction_purpose', '')
        doc_type = getattr(st.session_state, 'document_type', '')
        custom = getattr(st.session_state, 'custom_instructions', '')
        
        extraction_config['purpose_of_extraction'] = purpose.strip() if purpose and purpose.strip() else ""
        extraction_config['document_type'] = f"{doc_type.strip()}" if doc_type and doc_type.strip() else ""
        extraction_config['additional_instructions'] = custom.strip() if custom and custom.strip() else ""
        
        # Debug: log what we're saving
        import logging
        logging.info(f"Saving extraction context: purpose='{purpose}', doc_type='{doc_type}', custom='{custom}'")
        
        # Save updated config back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Log the error for debugging
        import logging
        logging.error(f"Error saving extraction context: {e}")
        # Don't raise the error to avoid breaking the extraction flow

def build_additional_instructions() -> str:
    """Build combined additional instructions from the three components"""
    instructions = []
    
    # Add structured purpose and document type instruction
    if st.session_state.extraction_purpose.strip() and st.session_state.document_type.strip():
        purpose_instruction = (
            f"The purpose of this extraction task is {st.session_state.extraction_purpose.strip()}. "
            f"Therefore, the document should be related to {st.session_state.document_type.strip()}. "
            f"Do not attempt to extract data from non-related documents. "
            f"If the documents are not related, output 'n/a' for all fields."
        )
        instructions.append(purpose_instruction)
    
    # Add custom instructions if provided
    if st.session_state.custom_instructions.strip():
        instructions.append(st.session_state.custom_instructions.strip())
    
    return '\n\n'.join(instructions)

def validate_azure_configuration() -> tuple[bool, str, str]:
    """Validate Azure configuration and return status, message, and CSS class"""
    use_azure = st.session_state.get('use_azure', False)
    generation_model = st.session_state.get('model_generation_model', 'claude-sonnet-4-20250514')
    extraction_model = st.session_state.get('extraction_model', 'gpt-4.1-2025-04-14')
    
    if not use_azure:
        return True, "", ""
    
    # Check if models are OpenAI
    is_generation_openai = generation_model and 'gpt' in generation_model.lower()
    is_extraction_openai = extraction_model and 'gpt' in extraction_model.lower()
    
    # Case 1: Both OpenAI models with Azure
    if is_generation_openai and is_extraction_openai:
        return True, "🔒 Using AZURE endpoint for data model generation and extraction", "certificate-green"
    
    # Case 2: Claude for generation, OpenAI for extraction with Azure
    elif not is_generation_openai and is_extraction_openai:
        return True, "🔒 Using AZURE endpoint for knowledge extraction only", "certificate-green"
    
    # Case 3: Claude for both generation and extraction with Azure (blocked)
    elif not is_generation_openai and not is_extraction_openai:
        return False, "🚫 This application does not support CLAUDE with AZURE endpoint. Select OpenAI model for extraction task to proceed with AZURE endpoint for knowledge extraction", "certificate-red"
    
    # Case 4: OpenAI for generation, Claude for extraction with Azure (blocked)
    elif is_generation_openai and not is_extraction_openai:
        return False, "🚫 This application does not support CLAUDE with AZURE endpoint. Select OpenAI model for extraction task to proceed with AZURE endpoint for knowledge extraction", "certificate-red"
    
    return True, "", ""

def parse_additional_instructions(combined_instructions: str):
    """Parse combined additional instructions back into components"""
    if not combined_instructions.strip():
        return "", "", ""
    
    # Split by double newlines to get separate instruction blocks
    instruction_blocks = combined_instructions.split('\n\n')
    
    extraction_purpose = ""
    document_type = ""
    custom_instructions = ""
    
    for block in instruction_blocks:
        # Check if this block contains the structured purpose/document type instruction
        if "The purpose of this extraction task is" in block and "Therefore, the document should be related to" in block:
            # Extract purpose and document type from the structured instruction
            import re
            purpose_match = re.search(r"The purpose of this extraction task is (.+?)\. Therefore, the document should be related to", block)
            doc_type_match = re.search(r"Therefore, the document should be related to (.+?)\. Do not attempt", block)
            
            if purpose_match:
                extraction_purpose = purpose_match.group(1).strip()
            if doc_type_match:
                document_type = doc_type_match.group(1).strip()
        else:
            # This is a custom instruction block
            if custom_instructions:
                custom_instructions += "\n\n" + block.strip()
            else:
                custom_instructions = block.strip()
    
    return extraction_purpose, document_type, custom_instructions

def load_saved_models() -> List[Dict[str, str]]:
    """Load list of saved model configurations from templates folders"""
    models = []
    use_cases_dir = "templates"
    
    if os.path.exists(use_cases_dir):
        for use_case_folder in os.listdir(use_cases_dir):
            folder_path = os.path.join(use_cases_dir, use_case_folder)
            if os.path.isdir(folder_path):
                # Look for config.json in each use case folder
                config_file = os.path.join(folder_path, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            extraction_config = config_data.get('extraction_config', {})
                            
                            # Calculate field count based on extraction type
                            field_count = 0
                            extraction_type = extraction_config.get('extraction_type', 'single_type')
                            
                            if extraction_type == 'multi_type_with_relationships':
                                # Case 2: Count fields from all stages
                                strategy = extraction_config.get('extraction_strategy', {})
                                if strategy and strategy.get('stages'):
                                    stages = strategy.get('stages', [])
                                    for stage in stages:
                                        field_count += len(stage.get('extraction_fields', []))
                                else:
                                    # Fallback: Count fields from parsed_fields if strategy is null/empty
                                    field_count = len(extraction_config.get('parsed_fields', []))
                            else:
                                # Single-type or Case 1: Count fields from fields array
                                field_count = len(extraction_config.get('fields', []))
                            
                            models.append({
                                'folder': use_case_folder,
                                'use_case': extraction_config.get('use_case', use_case_folder),
                                'description': extraction_config.get('description', 'No description'),
                                'model_name': extraction_config.get('main_model_name', 'Unknown'),
                                'field_count': field_count,
                                'extraction_type': extraction_type,
                                'created_at': extraction_config.get('created_at', 'Unknown'),
                                'has_config': True
                            })
                    except Exception as e:
                        # Skip invalid config files
                        continue
                else:
                    # Include use-case folders missing config.json so they still show up
                    models.append({
                        'folder': use_case_folder,
                        'use_case': use_case_folder,
                        'description': 'No config.json found in this use-case folder',
                        'model_name': 'Unknown',
                        'field_count': 0,
                        'created_at': 'Unknown',
                        'has_config': False
                    })
    
    return models

def save_model_config(config_data: Dict[str, Any], use_case_name: str):
    """Save model configuration to use-case folder"""
    config_path = get_use_case_path(use_case_name, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    return config_path

def load_model_config(relative_path: str) -> Dict[str, Any]:
    """Load model configuration from use-case folder"""
    full_path = os.path.join("templates", relative_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_field_form(field_index: int, field_data: Optional[Dict] = None):
    """Create a compact form for a single field configuration"""
    
    with st.container():
        st.markdown(f'<div class="field-container">', unsafe_allow_html=True)
        
        col_header, col_remove = st.columns([4, 1])
        with col_header:
            st.markdown(f"**📝 Field {field_index + 1}**")
        with col_remove:
            if st.button("🗑️", key=f"remove_{field_index}", help="Remove field"):
                st.session_state.fields.pop(field_index)
                st.rerun()
        
        # Four equal-width columns for the four attributes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            field_name = st.text_input(
                "Field Name",
                value=field_data.get('field_name', '') if field_data else '',
                key=f"field_name_{field_index}",
                help="Name of the field to extract"
            )
        
        with col2:
            field_description = st.text_area(
                "Field Description",
                value=field_data.get('description', '') if field_data else '',
                key=f"description_{field_index}",
                height=80,
                help="What information to extract"
            )
        
        with col3:
            # Categories/Classes (optional)
            categories_input = st.text_area(
                "Categories/Classes (Optional)",
                value='\n'.join(field_data.get('enum_values', [])) if field_data and field_data.get('enum_values') else '',
                key=f"categories_{field_index}",
                height=80,
                help="One category per line for classification"
            )
            categories = [cat.strip() for cat in categories_input.split('\n') if cat.strip()] if categories_input.strip() else None
        
        with col4:
            # Determine if this should be enum based on categories
            available_types = ['str', 'int', 'float', 'bool', 'list[str]']
            if categories:
                available_types.extend(['enum', 'list[enum]'])
            
            current_type = field_data.get('field_type', 'str') if field_data else 'str'
            if current_type not in available_types:
                current_type = 'str'
            
            field_type = st.selectbox(
                "Data Type",
                options=available_types,
                index=available_types.index(current_type),
                key=f"field_type_{field_index}",
                help="Choose appropriate data type"
            )
        
        # Required checkbox (full width)
        required = st.checkbox(
            "Required Field",
            value=field_data.get('required', True) if field_data else True,
            key=f"required_{field_index}"
        )
        
        # Update session state
        st.session_state.fields[field_index] = {
            'field_name': field_name,
            'field_type': field_type,
            'description': field_description,
            'required': required,
            'enum_values': categories
        }
        
        st.markdown('</div>', unsafe_allow_html=True)

def validate_configuration() -> tuple[bool, List[str]]:
    """Validate the current configuration"""
    errors = []

    if not st.session_state.use_case.strip():
        errors.append("Use case name is required")

    if not st.session_state.main_model_name.strip():
        errors.append("Main model name is required")

    # Validate based on configuration mode
    if st.session_state.configuration_mode == "text_description":
        # For text description mode, validate text description and parsed fields
        if not st.session_state.text_description.strip():
            errors.append("Text description is required")
        elif len(st.session_state.text_description.strip()) < 20:
            errors.append("Text description should be at least 20 characters long")
        elif len(st.session_state.text_description.strip()) > 2000:
            errors.append("Text description should be less than 2000 characters")
        
        if not st.session_state.fields and not st.session_state.parsed_fields:
            errors.append("Please parse the description to extract fields first")
    else:
        # For field-by-field mode, validate individual fields
        if not st.session_state.fields:
            errors.append("At least one field is required")

        field_names = []
        for i, field in enumerate(st.session_state.fields):
            if not field.get('field_name', '').strip():
                errors.append(f"Field {i+1}: Field name is required")
            else:
                if field['field_name'] in field_names:
                    errors.append(f"Field {i+1}: Duplicate field name '{field['field_name']}'")
                field_names.append(field['field_name'])

            if not field.get('description', '').strip():
                errors.append(f"Field {i+1}: Description is required")

            if field.get('field_type') in ['enum', 'list[enum]'] and not field.get('enum_values'):
                errors.append(f"Field {i+1}: Categories are required for enum types")

    return len(errors) == 0, errors

def export_configuration() -> Dict[str, Any]:
    """Export the current configuration"""
    extraction_type = st.session_state.get('extraction_type', 'single_type')
    
    # Base configuration
    config = {
        'extraction_config': {
            'use_case': st.session_state.use_case,
            'description': st.session_state.description,
            'main_model_name': st.session_state.main_model_name,
            'purpose_of_extraction': st.session_state.extraction_purpose.strip() if st.session_state.extraction_purpose.strip() else "",
            'document_type': f"{st.session_state.document_type.strip()}" if st.session_state.document_type.strip() else "",
            'additional_instructions': st.session_state.custom_instructions.strip() if st.session_state.custom_instructions.strip() else "",
            'created_at': datetime.now().isoformat(),
            'extraction_type': extraction_type
        }
    }
    
    # Generate config structure based on extraction type
    if extraction_type == 'multi_type_with_relationships':
        # Case 2: Generate hierarchical extraction config structure
        config['extraction_config']['configuration_mode'] = 'text_description'
        config['extraction_config']['text_description'] = st.session_state.text_description
        config['extraction_config']['parsed_fields'] = st.session_state.parsed_fields
        config['extraction_config']['description_parsing_model'] = st.session_state.description_parsing_model
        config['extraction_config']['parsing_method'] = st.session_state.get('parsing_method', 'fast')
        
        # For Case 2, we need to generate the extraction_strategy
        # This will be populated when the Case 2 orchestrator creates the use case
        # For now, we'll leave it empty - it will be filled by Case2Config.create_config()
        config['extraction_config']['extraction_strategy'] = None
        
    else:
        # Single-type or Case 1: Generate traditional config structure
        fields_config = []
        for field in st.session_state.fields:
            field_config = {
                'field_name': field['field_name'],
                'field_type': field['field_type'],
                'description': field['description'],
                'required': field['required'],
                'enum_values': field['enum_values'] if field.get('enum_values') else None
            }
            fields_config.append(field_config)
        
        config['extraction_config']['fields'] = fields_config
        
        # Add text description mode specific information
        if st.session_state.configuration_mode == "text_description":
            config['extraction_config']['configuration_mode'] = 'text_description'
            config['extraction_config']['text_description'] = st.session_state.text_description
            config['extraction_config']['parsed_fields'] = st.session_state.parsed_fields
            config['extraction_config']['description_parsing_model'] = st.session_state.description_parsing_model
        else:
            config['extraction_config']['configuration_mode'] = 'fields'
        
        # Add parsing method to configuration
        config['extraction_config']['parsing_method'] = st.session_state.get('parsing_method', 'fast')

    return config

def configuration_section():
    """Configuration section of the UI"""
    st.markdown('<div class="section-header">🔧 Use Case Configuration</div>', unsafe_allow_html=True)
    
    # Ensure templates folder exists
    ensure_use_cases_folder()
    
    # Model selection or creation with enhanced display
    saved_models = load_saved_models()
    
    if saved_models:
        st.markdown("**📋 Available Use Cases**")
        
        # Create selection options with detailed descriptions
        model_options = ["🆕 Create New Use Case"]
        model_display_names = {}
        
        for model in saved_models:
            missing = " — missing config" if not model.get('has_config', True) else ""
            extraction_type = model.get('extraction_type', 'single_type')
            type_icon = "🔗" if extraction_type == 'multi_type_with_relationships' else "📄" if extraction_type == 'multi_type' else "📄"
            display_name = f"{type_icon} {model['use_case']} ({model['model_name']}) - {model['field_count']} fields{missing}"
            model_options.append(display_name)
            model_display_names[display_name] = model
        
        selected_option = st.selectbox(
            "Choose an extraction use case to load or create new",
            model_options,
            key="model_selection"
        )
        
        if selected_option != "🆕 Create New Use Case":
            selected_model = model_display_names[selected_option]
            
            # Show model details in an expandable section
            with st.expander(f"📋 Model Details: {selected_model['use_case']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Use Case:** {selected_model['use_case']}")
                    st.write(f"**Model Name:** {selected_model['model_name']}")
                    st.write(f"**Fields:** {selected_model['field_count']}")
                with col2:
                    st.write(f"**Description:** {selected_model['description'][:100]}...")
                    st.write(f"**Created:** {selected_model['created_at'][:10] if selected_model['created_at'] != 'Unknown' else 'Unknown'}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("📁 Load Selected Model", type="primary"):
                    try:
                        if not selected_model.get('has_config', True):
                            st.warning("This use case doesn't have a config.json yet. Open it and click 'Save Model' to create one.")
                            return
                        config_path = f"{selected_model['folder']}/config.json"
                        config_data = load_model_config(config_path)
                        extraction_config = config_data['extraction_config']
                        
                        st.session_state.use_case = extraction_config.get('use_case', '')
                        st.session_state.description = extraction_config.get('description', '')
                        
                        # Load extraction type (important for multi-type vs single-type)
                        extraction_type = extraction_config.get('extraction_type', 'single_type')
                        st.session_state.extraction_type = extraction_type
                        
                        # Debug logging for extraction type
                        logger.info(f"Loaded use case '{st.session_state.use_case}' with extraction_type: {extraction_type}")

                        # Handle different extraction types
                        if extraction_type == 'multi_type_with_relationships':
                            # Case 2: Load from extraction_strategy
                            strategy = extraction_config.get('extraction_strategy', {})
                            if strategy and strategy.get('stages'):
                                stages = strategy.get('stages', [])
                                
                                # Generate model name from use case name
                                st.session_state.main_model_name = f"{st.session_state.use_case}Model"
                                
                                # Extract fields from all stages for display
                                all_fields = []
                                for stage in stages:
                                    stage_fields = stage.get('extraction_fields', [])
                                    for field in stage_fields:
                                        all_fields.append({
                                            'field_name': field.get('field_name', ''),
                                            'field_type': field.get('field_type', 'str'),
                                            'description': field.get('description', ''),
                                            'required': field.get('required', True),
                                            'stage': stage.get('stage_name', '')
                                        })
                                
                                st.session_state.fields = all_fields
                                st.session_state.configuration_mode = 'text_description'  # Case 2 always uses text description
                                st.session_state.text_description = extraction_config.get('text_description', '')
                                st.session_state.parsed_fields = all_fields
                            else:
                                # Fallback: Load from parsed_fields if strategy is null/empty
                                st.session_state.main_model_name = extraction_config.get('main_model_name', f"{st.session_state.use_case}Model")
                                st.session_state.configuration_mode = 'text_description'
                                st.session_state.text_description = extraction_config.get('text_description', '')
                                st.session_state.parsed_fields = extraction_config.get('parsed_fields', [])
                                st.session_state.fields = extraction_config.get('parsed_fields', [])
                            
                        else:
                            # Single-type or Case 1: Load traditional fields
                            st.session_state.main_model_name = extraction_config.get('main_model_name', '')
                            
                            # Load configuration mode
                            config_mode = extraction_config.get('configuration_mode', 'fields')
                            st.session_state.configuration_mode = config_mode

                            # Load text description mode specific data
                            if config_mode == 'text_description':
                                st.session_state.text_description = extraction_config.get('text_description', '')
                                st.session_state.parsed_fields = extraction_config.get('parsed_fields', [])
                                st.session_state.description_parsing_model = extraction_config.get('description_parsing_model', 'claude-sonnet-4-20250514')
                                # Use parsed fields as current fields
                                st.session_state.fields = st.session_state.parsed_fields
                            else:
                                # Load field-by-field mode data
                                st.session_state.fields = extraction_config.get('fields', [])
                        
                        # Load parsing method
                        st.session_state.parsing_method = extraction_config.get('parsing_method', 'fast')

                        # Handle both old format (combined additional_instructions) and new format (three separate fields)
                        if 'purpose_of_extraction' in extraction_config or 'document_type' in extraction_config:
                            # New format with separate fields
                            st.session_state.extraction_purpose = extraction_config.get('purpose_of_extraction', '')
                            # Remove the appended warning text from document_type
                            doc_type_raw = extraction_config.get('document_type', '')
                            if ". Do not attempt to extract data from non-related documents" in doc_type_raw:
                                st.session_state.document_type = doc_type_raw.split(". Do not attempt to extract data from non-related documents")[0]
                            else:
                                st.session_state.document_type = doc_type_raw
                            st.session_state.custom_instructions = extraction_config.get('additional_instructions', '')
                        else:
                            # Old format with combined additional_instructions - parse it
                            combined_instructions = extraction_config.get('additional_instructions', '')
                            purpose, doc_type, custom = parse_additional_instructions(combined_instructions)
                            st.session_state.extraction_purpose = purpose
                            st.session_state.document_type = doc_type
                            st.session_state.custom_instructions = custom

                        # Build combined instructions for backward compatibility
                        st.session_state.additional_instructions = build_additional_instructions()
                        
                        st.success(f"✅ Loaded model: {selected_model['use_case']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
            with col2:
                if st.button("🗑️ Delete Model"):
                    try:
                        folder_path = f"templates/{selected_model['folder']}"
                        import shutil
                        shutil.rmtree(folder_path)
                        st.success(f"✅ Deleted model: {selected_model['use_case']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting model: {str(e)}")
        
        st.markdown("---")
    else:
        st.info("💡 No existing models found. Create your first model below.")
    
    # Basic configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.use_case = st.text_input(
            "Use Case Name",
            value=st.session_state.use_case,
            help="Name for your extraction use case"
        )
        
        st.session_state.main_model_name = st.text_input(
            "Model Name",
            value=st.session_state.main_model_name,
            help="Name for the generated model class"
        )
    
    with col2:
        st.session_state.description = st.text_area(
            "Description",
            value=st.session_state.description,
            height=100,
            help="What you're extracting and why"
        )
        
        if st.button("🔤 Auto-Generate Model Name"):
            if st.session_state.use_case:
                suggested = st.session_state.use_case.replace(' ', '').replace('-', '').replace('_', '')
                if not suggested.endswith('Info'):
                    suggested += 'Info'
                st.session_state.main_model_name = suggested
                st.rerun()

    # Extraction Type Selection
    st.markdown("**📋 Extraction Type**")
    
    extraction_type = st.radio(
        "Choose how to process your documents:",
        ["Single-type documents", "Multi-type documents without relationships", "Multi-type documents with relationships"],
        index=0 if st.session_state.extraction_type == "single_type" else 
              1 if st.session_state.extraction_type == "multi_type" else 2,
        help="Select whether you're processing documents of the same type, diverse document types, or documents with hierarchical relationships"
    )
    
    # Update session state
    if extraction_type == "Single-type documents":
        st.session_state.extraction_type = "single_type"
    elif extraction_type == "Multi-type documents without relationships":
        st.session_state.extraction_type = "multi_type"
    else:  # Multi-type documents with relationships
        st.session_state.extraction_type = "multi_type_with_relationships"
    
    st.markdown("---")
    
    # Configuration Mode Selection (conditional based on extraction type)
    if st.session_state.extraction_type == "single_type":
        st.markdown("**⚙️ Configuration Mode**")
        
        col1, col2 = st.columns(2)
        with col1:
            configuration_mode = st.radio(
                "Choose how to configure your extraction fields:",
                ["Field-by-Field Configuration", "Text Description"],
                index=0 if st.session_state.configuration_mode == "fields" else 1,
                help="Select how you want to define what information to extract"
            )
            st.session_state.configuration_mode = configuration_mode
        
        with col2:
            # This column is now empty - the description parsing model is moved to the text description section
            pass
    elif st.session_state.extraction_type == "multi_type":
        # For multi-type documents, only show text description mode
        st.markdown("**⚙️ Multi-Document Configuration**")
        st.info("💡 Multi-type document extraction uses Text Description mode for intelligent document analysis and routing.")
        
        configuration_mode = "Text Description"
        st.session_state.configuration_mode = "text_description"
    else:  # multi_type_with_relationships (Case 2)
        # For Case 2 hierarchical extraction
        st.markdown("**⚙️ Hierarchical Extraction Configuration**")
        st.info("🔗 Hierarchical extraction uses Text Description mode to identify document relationships and create multi-stage extraction workflows.")
        
        configuration_mode = "Text Description"
        st.session_state.configuration_mode = "text_description"

    # Update configuration mode in session state
    if st.session_state.extraction_type == "single_type":
        if configuration_mode == "Field-by-Field Configuration":
            st.session_state.configuration_mode = "fields"
        else:
            st.session_state.configuration_mode = "text_description"
    else:
        # Multi-type always uses text description
        st.session_state.configuration_mode = "text_description"

    st.markdown("---")

    # Text Description Interface
    if configuration_mode == "Text Description":
        if st.session_state.extraction_type == "multi_type":
            st.markdown("**📝 Multi-Document Extraction Description**")
            st.info("🔍 Describe what information to extract from your diverse documents. The system will automatically classify documents and route them to appropriate extraction pipelines.")
        elif st.session_state.extraction_type == "multi_type_with_relationships":
            st.markdown("**📝 Hierarchical Multi-Document Extraction Description**")
            st.info("🔗 Describe how your documents are related and what information flows between them. The system will create a multi-stage extraction workflow.")
            
            # Show relationship examples for Case 2
            with st.expander("💡 Relationship Examples"):
                st.markdown("""
                **Order Processing:**
                - Orders → Product Specs (via product_id) → Pricing (via product_id)
                
                **Customer Management:**
                - Customer Info → Orders (via customer_id) → Payments (via order_id)
                
                **Project Management:**
                - Projects → Tasks (via project_id) → Resources (via task_id)
                
                **Supply Chain:**
                - Purchase Orders → Shipments (via po_id) → Invoices (via shipment_id)
                """)
        else:
            st.markdown("**📝 Text Description Interface**")
        
        # AI Model for Parsing Description selection
        st.markdown("**🤖 AI Model for Parsing Description**")
        selected_description_model = st.selectbox(
            "Choose the AI model to parse your text description:",
            options=['claude-sonnet-4-20250514', 'gpt-4.1-2025-04-14'],
            index=0 if st.session_state.description_parsing_model == 'claude-sonnet-4-20250514' else 1,
            help="Choose the AI model to parse your text description",
            key="description_parsing_model_selectbox"
        )
        st.session_state.description_parsing_model = selected_description_model
        st.markdown("---")

        # Example descriptions dropdown
        col1, col2 = st.columns([3, 1])
        with col1:
            examples = get_example_descriptions()
            example_choice = st.selectbox(
                "Choose an example description to get started:",
                ["Custom Description"] + list(examples.keys()),
                help="Select an example or write your own description"
            )
        
        with col2:
            if st.button("📋 Load Example"):
                if example_choice != "Custom Description":
                    st.session_state.text_description = examples[example_choice]
                    st.rerun()
        
        # Text description input
        if st.session_state.extraction_type == "multi_type":
            st.session_state.text_description = st.text_area(
                "Describe what information you want to extract from your diverse documents:",
                value=st.session_state.text_description,
                height=150,
                help="Describe in natural language what fields and information you want to extract from different types of documents. The system will automatically classify documents and extract the appropriate information from each type.",
                placeholder="Example: Extract customer information from invoices (customer name, invoice number, amount), product specifications from technical documents (product name, specifications, dimensions), and order details from purchase orders (order number, items, quantities)..."
            )
        elif st.session_state.extraction_type == "multi_type_with_relationships":
            st.session_state.text_description = st.text_area(
                "Describe how your documents are related and what information flows between them:",
                value=st.session_state.text_description,
                height=150,
                help="Describe the hierarchical relationships between your documents. Explain how information flows from one document type to another using common keys or identifiers.",
                placeholder="Example: Extract order information from orders (order_id, customer_id, product_id), then match with product specifications using product_id to get dimensions and specs, then get pricing from price list using the same product_id to find unit price and currency..."
            )
        else:
            st.session_state.text_description = st.text_area(
                "Describe what information you want to extract:",
                value=st.session_state.text_description,
                height=150,
                help="Describe in natural language what fields and information you want to extract from documents",
                placeholder="Example: Extract company information including name, industry, revenue, number of employees, and key executives from business documents..."
            )
        
        # Character counter and validation
        char_count = len(st.session_state.text_description)
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if char_count > 0:
                if char_count < 20:
                    st.warning("⚠️ Description should be at least 20 characters")
                elif char_count > 2000:
                    st.error("❌ Description should be less than 2000 characters")
                else:
                    st.success(f"✅ {char_count} characters")
        
        with col2:
            if st.button("🔍 Parse Description", disabled=st.session_state.parsing_in_progress):
                if parse_text_description():
                    st.success("✅ Description parsed successfully!")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Clear"):
                st.session_state.text_description = ""
                st.session_state.parsed_fields = []
                st.session_state.fields = []
                st.rerun()
        
        # Show parsing tips
        with st.expander("💡 Tips for writing effective descriptions"):
            tips = get_parsing_tips()
            for tip in tips:
                st.markdown(f"• {tip}")
        
        # Show parsed fields preview
        if st.session_state.parsed_fields:
            st.markdown("**👀 Parsed Fields Preview**")
            df = pd.DataFrame(st.session_state.parsed_fields)
            st.dataframe(df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Use Parsed Fields"):
                    st.session_state.fields = st.session_state.parsed_fields
                    st.success("✅ Fields updated successfully!")
                    st.rerun()
            
            with col2:
                if st.button("✏️ Edit Fields"):
                    # Switch to field-by-field mode and populate fields with parsed data
                    st.session_state.configuration_mode = "fields"
                    st.session_state.fields = st.session_state.parsed_fields.copy()
                    st.success("✅ Switched to field-by-field mode. You can now edit individual fields below.")
                    st.rerun()
        
        st.markdown("---")

    # Model Selection Settings
    st.markdown("**⚙️ Model Settings**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Initialize session state for model settings if not exists
        if 'model_generation_model' not in st.session_state:
            st.session_state.model_generation_model = 'claude-sonnet-4-20250514'
        if 'extraction_model' not in st.session_state:
            st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
        if 'use_azure' not in st.session_state:
            st.session_state.use_azure = False

        selected_generation_model = st.selectbox(
            "Model for Data Model Generation",
            options=['claude-sonnet-4-20250514', 'gpt-4.1-2025-04-14'],
            index=0 if st.session_state.model_generation_model == 'claude-sonnet-4-20250514' else 1,
            help="Choose the AI model to generate Pydantic data models from your field configurations",
            key="model_generation_selectbox"
        )
        st.session_state.model_generation_model = selected_generation_model

    with col2:
        # Extraction model options depend on Azure selection
        azure_enabled = st.session_state.get('use_azure', False)
        if azure_enabled:
            extraction_options = ['gpt-4.1-2025-04-14']
            default_extraction = 'gpt-4.1-2025-04-14'
        else:
            extraction_options = ['gpt-4.1-2025-04-14', 'claude-sonnet-4-20250514']
            default_extraction = st.session_state.extraction_model

        if default_extraction not in extraction_options:
            default_extraction = extraction_options[0]

        selected_extraction_model = st.selectbox(
            "Extraction Model",
            options=extraction_options,
            index=extraction_options.index(default_extraction),
            help="Choose the AI model for extracting data from documents",
            disabled=azure_enabled and st.session_state.extraction_model != 'gpt-4.1-2025-04-14',
            key="extraction_model_selectbox"
        )
        st.session_state.extraction_model = selected_extraction_model

        # Show disabled message if Azure is enabled and Claude was selected
        if azure_enabled and st.session_state.extraction_model and 'claude' in st.session_state.extraction_model.lower():
            st.warning("⚠️ This application uses only OpenAI model with Azure endpoint for extraction task")

    with col3:
        st.session_state.use_azure = st.checkbox(
            "Use Microsoft Azure Endpoint",
            value=st.session_state.get('use_azure', False),
            help="Select for secure data processing. Requires MS AZURE API key"
        )

    with col4:
        # This column is now empty - parsing method moved to Extraction section
        pass
    
    # Force GPT-4.1 selection when Azure is enabled
    if st.session_state.use_azure and st.session_state.extraction_model != 'gpt-4.1-2025-04-14':
        st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
        st.rerun()
    
    # Azure Configuration Validation and Certificate Display
    is_valid, message, css_class = validate_azure_configuration()
    if message:
        st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)
    
    # Fields configuration (only show in field-by-field mode)
    if st.session_state.configuration_mode == "fields":
        st.markdown("**📋 Extraction Fields**")
    
        col_add, col_info = st.columns([1, 4])
        with col_add:
            if st.button("➕ Add Field", type="primary", key="add_field_btn"):
                new_field = {
                    'field_name': '',
                    'field_type': 'str',
                    'description': '',
                    'required': True,
                    'enum_values': None
                }
                st.session_state.fields.append(new_field)
                st.rerun()

        with col_info:
            if st.session_state.fields:
                st.markdown(f"*Currently configured: {len(st.session_state.fields)} fields*")
            else:
                st.markdown("*No fields defined yet*")

        # Display fields
        if st.session_state.fields:
            for i, field in enumerate(st.session_state.fields):
                create_field_form(i, field)

    # Validation and save (show for both modes)
    is_valid, errors = validate_configuration()

    if errors:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**⚠️ Configuration Issues:**")
        for error in errors:
            st.markdown(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)

    if is_valid:
        st.markdown('<div class="success-box">✅ Configuration is ready!</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save Model", type="primary"):
                try:
                    config = export_configuration()
                    config_path = save_model_config(config, st.session_state.use_case)
                    use_case_folder = os.path.dirname(config_path)
                    st.success(f"✅ Model saved to: {use_case_folder}")
                except Exception as e:
                    st.error(f"❌ Error saving: {str(e)}")

        with col2:
            config = export_configuration()
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="📥 Download",
                data=config_json,
                file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_config.json",
                mime="application/json"
            )

        with col3:
            if st.button("🚀 Go to Extraction", type="secondary"):
                load_extraction_context_from_current_config()  # Load context before switching
                st.session_state.current_tab = "Extraction"
                st.rerun()

def extraction_section():
    """Extraction section of the UI"""
    st.markdown('<div class="section-header">🎯 Data Extraction</div>', unsafe_allow_html=True)
    
    # Check if configuration is ready
    is_valid, _ = validate_configuration()
    
    if not is_valid:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**⚠️ Please complete model configuration first**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("← Back to Configuration"):
            st.session_state.current_tab = "Configuration"
            st.rerun()
        return
    
    # Extraction context and instructions
    st.markdown("**📋 Extraction Context**")
    st.info("💡 **Enhance your extraction quality!** Providing context about your extraction purpose and document types helps the AI understand your specific needs and deliver more accurate results.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.extraction_purpose = st.text_input(
            "Purpose of Extraction",
            value=st.session_state.extraction_purpose,
            help="e.g., to extract the information against key fields from AI consultancy document(s)",
            placeholder="e.g., to extract information from AI consultancy reports"
        )
    
    with col2:
        st.session_state.document_type = st.text_input(
            "Type of Document(s)",
            value=st.session_state.document_type,
            help="e.g., Documents for AI consultancy to companies",
            placeholder="e.g., AI consultancy reports, business documents"
        )
    
    # Fields are now optional - no validation errors needed
    
    # Additional custom instructions (optional)
    default_custom_instructions = ""
    if 'custom_instructions' not in st.session_state or not st.session_state.custom_instructions:
        st.session_state.custom_instructions = default_custom_instructions
    
    st.session_state.custom_instructions = st.text_area(
        "Additional Custom Instructions (Optional)",
        value=st.session_state.custom_instructions,
        help="Additional specific instructions for the extraction process",
        placeholder="e.g., Focus on the first page only, ignore footnotes, use specific date formats"
    )
    
    # File selection
    st.markdown("**📁 Document Selection**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selection_mode = st.radio(
            "Selection Mode",
            ["Select Folder", "Select Individual Files"],
            key="selection_mode"
        )
    
    with col2:
        if selection_mode == "Select Folder":
            col_input, col_browse = st.columns([3, 1])
            
            # Initialize folder path in session state if not exists
            if 'selected_folder_path' not in st.session_state:
                st.session_state.selected_folder_path = ""
            
            with col_input:
                folder_path = st.text_input(
                    "Folder Path",
                    value=st.session_state.selected_folder_path,
                    help="Path to folder containing documents",
                    key="folder_path_input"
                )
                # Update session state when user types
                if folder_path != st.session_state.selected_folder_path:
                    st.session_state.selected_folder_path = folder_path
            
            with col_browse:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                if st.button("📁 Browse", help="Open folder selection dialog"):
                    # Use tkinter for folder selection dialog
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        
                        # Create a root window and hide it
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        
                        # Open folder dialog
                        selected_folder = filedialog.askdirectory(
                            title="Select Folder Containing Documents"
                        )
                        
                        # Destroy the root window
                        root.destroy()
                        
                        if selected_folder:
                            st.session_state.selected_folder_path = selected_folder
                            st.rerun()
                    except ImportError:
                        st.error("❌ Folder dialog not available. Please enter path manually.")
                    except Exception as e:
                        st.error(f"❌ Error opening folder dialog: {str(e)}")
            
            # Use the session state value for folder processing
            folder_path = st.session_state.selected_folder_path
            
            if folder_path and os.path.exists(folder_path):
                supported_files = []
                for ext in ['.pdf', '.docx', '.doc']:
                    supported_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
                
                if supported_files:
                    st.success(f"✅ Found {len(supported_files)} supported documents")
                    st.session_state.selected_files = supported_files
                    
                    # Show preview of found files
                    with st.expander(f"📋 Preview Files ({len(supported_files)} files)"):
                        for file_path in supported_files[:10]:  # Show first 10
                            st.text(f"📄 {os.path.basename(file_path)}")
                        if len(supported_files) > 10:
                            st.text(f"... and {len(supported_files) - 10} more files")
                else:
                    st.warning("No supported documents (.pdf, .docx, .doc) found in folder")
            elif folder_path:
                st.error("❌ Folder path does not exist")
        else:
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'doc'],
                accept_multiple_files=True,
                help="Select one or more documents to process"
            )
            
            if uploaded_files and not st.session_state.get('files_processed', False):
                # Only process files if they haven't been processed yet
                # Store files in session state as BytesIO objects to avoid temp files
                file_data = []
                for uploaded_file in uploaded_files:
                    file_data.append({
                        'name': uploaded_file.name,
                        'content': uploaded_file.getbuffer(),
                        'type': uploaded_file.type
                    })
                
                st.session_state.uploaded_file_data = file_data
                st.session_state.selected_files = [f"memory_{f['name']}" for f in file_data]  # Placeholder paths
                st.session_state.files_processed = True  # Mark files as processed
                st.success(f"✅ Selected {len(file_data)} documents (stored in memory)")
    
    # Clear files button
    if st.session_state.selected_files:
        if st.button("🗑️ Clear Files", help="Clear selected files and clean up temporary files"):
            _cleanup_temp_files()
            st.success("✅ Files cleared and temporary files cleaned up")
            st.rerun()
    
    # Extraction execution
    if st.session_state.selected_files:
        st.markdown("**⚡ Run Extraction**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Documents", len(st.session_state.selected_files))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Fields", len(st.session_state.fields))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Model", st.session_state.main_model_name or "Not set")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Document Parsing Method Selection
        st.markdown("**📄 Document Parsing Method**")
        col_parse1, col_parse2 = st.columns([2, 1])
        
        with col_parse1:
            # Always show the parsing method selection
            parsing_options = ["Fast parsing"]
            parsing_labels = ["Fast parsing"]
            
            # Add Docling option if available
            if DOCLING_AVAILABLE:
                parsing_options.append("Docling parsing (slow without GPU)")
                parsing_labels.append("Docling parsing (slow without GPU)")
            else:
                # Show that Docling is not available
                parsing_options.append("Docling parsing (not installed)")
                parsing_labels.append("Docling parsing (not installed)")
            
            # Determine current index
            current_method = st.session_state.get('parsing_method', 'fast')
            if current_method == 'docling' and DOCLING_AVAILABLE:
                current_index = 1
            else:
                current_index = 0
                st.session_state.parsing_method = 'fast'  # Ensure it's set to fast if docling not available
            
            parsing_method_display = st.selectbox(
                "Choose parsing method:",
                options=parsing_options,
                index=current_index,
                help="Fast parsing uses PyMuPDF/docx libraries for all file types. Docling parsing provides advanced document structure analysis with CUDA acceleration when available.",
                key="parsing_method_selectbox"
            )
            
            # Update session state value based on display text
            if parsing_method_display == "Fast parsing":
                st.session_state.parsing_method = "fast"
            elif parsing_method_display == "Docling parsing (slow without GPU)":
                st.session_state.parsing_method = "docling"
            elif parsing_method_display == "Docling parsing (not installed)":
                st.warning("⚠️ **Docling parsing is not installed.** To use Docling parsing, install it with: `pip install docling`")
                st.session_state.parsing_method = "fast"  # Force fast parsing
        
        with col_parse2:
            current_method = st.session_state.get('parsing_method', 'fast')
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
                
                if format_option == "Markdown format":
                    st.session_state.docling_format = 'markdown'
                    st.caption("📝 **Markdown**: Clean text format - Use for letters, reports, articles, and general text extraction")
                else:
                    st.session_state.docling_format = 'structured'
                    st.caption("🏗️ **Structured**: Preserves layout - Use for forms, tables, multi-column documents, and structured data")
                    
            else:
                st.info("⚡ **Fast parsing** - Uses PyMuPDF/docx libraries for all file types")
                if not DOCLING_AVAILABLE:
                    st.warning("⚠️ Docling not available")
        
        # Rebuild models checkbox
        st.session_state.rebuild_models = st.checkbox(
            "🔧 Rebuild/Rebuild models", 
            value=st.session_state.rebuild_models,
            help="Rebuild models for a new use case, or rebuild existing model only when you have edited the existing model or modified additional instructions"
        )
        
        # Check Azure configuration validity before allowing extraction
        azure_valid, azure_message, azure_css = validate_azure_configuration()
        
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True, disabled=not azure_valid):
            if azure_valid:
                run_extraction()
            else:
                st.error("❌ Cannot start extraction due to invalid Azure configuration. Please check your model settings.")
    
    # Display results
    if st.session_state.extraction_results is not None:
        display_results()

def _extract_original_model_info(use_case_path: str) -> Dict[str, Any]:
    """Extract field names from the original single-type model for Case 2 consistency"""
    try:
        # Look for the original model file
        model_filename = f"{os.path.basename(use_case_path)}_models.py"
        model_path = os.path.join(use_case_path, model_filename)
        
        if os.path.exists(model_path):
            # Read the model file and extract field names
            with open(model_path, 'r', encoding='utf-8') as f:
                model_content = f.read()
            
            # Extract field names using regex
            import re
            field_pattern = r'(\w+):\s*\w+\s*=\s*Field\('
            field_names = re.findall(field_pattern, model_content)
            
            logger.info(f"Extracted original model field names: {field_names}")
            return {'field_names': field_names}
        
        return {}
        
    except Exception as e:
        logger.warning(f"Could not extract original model info: {e}")
        return {}

def run_extraction():
    """Execute the extraction process"""
    try:
        # Initialize messaging system
        messenger = get_messenger()
        
        # Get API configuration and model selections
        api_config = get_api_config()
        model_generation_model = st.session_state.get('model_generation_model', 'claude-sonnet-4-20250514')
        extraction_model = st.session_state.get('extraction_model', 'gpt-4.1-2025-04-14')
        
        # Ensure models are strings and not None
        if not model_generation_model:
            model_generation_model = 'claude-sonnet-4-20250514'
        if not extraction_model:
            extraction_model = 'gpt-4.1-2025-04-14'
        
        # Initialize components with model selections and API config
        # Pass API config for OpenAI model generation when Azure is enabled
        if model_generation_model and 'gpt' in model_generation_model.lower() and api_config.get('use_azure', False):
            model_generator = ModelGenerator(model_selection=model_generation_model, api_config=api_config)
        else:
            model_generator = ModelGenerator(model_selection=model_generation_model)
        
        # Initialize document parser based on selected method
        parsing_method = st.session_state.get('parsing_method', 'fast')
        if parsing_method == 'docling' and DOCLING_AVAILABLE:
            document_parser = DoclingParser()
        else:
            document_parser = DocumentParser()
            # Force fast parsing if docling is not available
            if parsing_method == 'docling' and not DOCLING_AVAILABLE:
                st.warning("⚠️ Docling parsing not available. Falling back to Fast parsing.")
                st.session_state.parsing_method = 'fast'
        
        # Initialize extractor based on extraction model selection
        if extraction_model and 'claude' in extraction_model.lower():
            # If Claude is selected for extraction (only when not using Azure)
            extractor = ClaudeExtractor(model_selection=extraction_model)
        else:
            # Use OpenAI for extraction
            extractor = OpenAIExtractor(api_config=api_config)
        
        # Generate models and prompts
        config = export_configuration()
        
        # Ensure config.json exists when starting extraction (covers text-based creation path)
        try:
            cfg_path = get_use_case_path(st.session_state.use_case, "config.json")
            if not os.path.exists(cfg_path):
                save_model_config(config, st.session_state.use_case)
        except Exception as _:
            pass
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Handle models (generate or load existing)
        use_case_name = st.session_state.use_case
        safe_use_case = use_case_name.replace(' ', '_').replace('-', '_')
        model_filename = f"{safe_use_case}_models.py"
        prompt_filename = f"{safe_use_case}_prompt.py"
        model_path = get_use_case_path(use_case_name, model_filename)
        prompt_path = get_use_case_path(use_case_name, prompt_filename)
        
        if st.session_state.rebuild_models:
            # Generate new models
            model_display_name = "Claude" if model_generation_model and 'claude' in model_generation_model.lower() else "OpenAI GPT-4.1"
            status_text.text(f"🔧 Generating new Pydantic models using {model_display_name}...")
            progress_bar.progress(10)
            try:
                # Generate models
                messenger.start_operation("Model Generation", f"Creating models for {len(st.session_state.fields)} fields")
                
                pydantic_model_class, model_code = model_generator.generate_models_from_config_data(config)
                
                messenger.complete_operation(True, f"Generated {len(st.session_state.fields)} field models")
                progress_bar.progress(20)
                
                status_text.text("💾 Saving generated models and prompts...")
                extraction_prompt = model_generator.get_extraction_prompt()
                
                # Save Python models and prompt to use-case folder
                model_generator.save_generated_models(model_path, model_code)
                model_generator.save_extraction_prompt(prompt_path)
                progress_bar.progress(30)
                
                status_text.text("✅ Models generated and saved successfully")
            except Exception as model_error:
                progress_bar.progress(0)
                status_text.text("❌ Model generation failed")
                st.error(f"❌ Model generation failed: {str(model_error)}")
                # Show the problematic configuration for debugging
                with st.expander("🔍 Debug Information"):
                    st.json(config)
                    st.text("If you see this error, try simplifying your field names and descriptions.")
                return
        else:
            # Load existing models
            status_text.text("📂 Loading existing models from saved files...")
            progress_bar.progress(10)
            try:
                if not os.path.exists(model_path) or not os.path.exists(prompt_path):
                    progress_bar.progress(0)
                    status_text.text("❌ Models not found")
                    st.error(f"❌ Models not found for use case '{use_case_name}'. Please check 'Build/Rebuild models' to generate them first.")
                    return
                
                # Load existing models and prompt
                pydantic_model_class, extraction_prompt = model_generator.load_models_and_prompt(model_path, prompt_path)
                progress_bar.progress(30)
                
                status_text.text("✅ Existing models loaded successfully")
            except Exception as load_error:
                progress_bar.progress(0)
                status_text.text("❌ Failed to load existing models")
                st.error(f"❌ Failed to load existing models: {str(load_error)}")
                st.info("💡 Try checking 'Rebuild models' to generate new models")
                return
        
        # Step 2: Parse documents
        # Show parsing method in status
        parsing_method = st.session_state.get('parsing_method', 'fast')
        parsing_label = "Docling parsing" if parsing_method == 'docling' else "Fast parsing"
        status_text.text(f"📄 {parsing_label} - Processing {len(st.session_state.selected_files)} document(s)...")
        progress_bar.progress(40)
        parsed_documents = []
        docling_chunks = []  # Store Docling chunks separately
        total_files = len(st.session_state.selected_files)
        
        # Debug: Log file selection details
        messenger.start_operation("Document Processing", f"{total_files} files selected")
        
        for i, file_path in enumerate(st.session_state.selected_files):
            messenger.debug(f"File {i+1}: {os.path.basename(file_path)}")
        
        # Handle both in-memory files and temp file paths
        uploaded_file_data = st.session_state.get('uploaded_file_data', [])
        
        for i, file_ref in enumerate(st.session_state.selected_files):
            temp_path_created = None
            try:
                # Determine if this is a memory reference or actual file path
                if file_ref.startswith("memory_"):
                    # Extract file from memory
                    file_name = file_ref[7:]  # Remove "memory_" prefix
                    file_data = next((f for f in uploaded_file_data if f['name'] == file_name), None)
                    
                    if not file_data:
                        st.warning(f"⚠️ Could not find file data for {file_name}")
                        continue
                    
                    # Create temporary file only for parsing
                    temp_path_created = f"temp_parsing_{file_data['name']}"
                    with open(temp_path_created, "wb") as f:
                        f.write(file_data['content'])
                    
                    actual_file_path = temp_path_created
                    file_basename = file_data['name']
                else:
                    # Regular file path
                    actual_file_path = file_ref
                    file_basename = os.path.basename(file_ref)
                
                messenger.document_processing(file_basename, parsing_method, total_files, i+1)
                
                if parsing_method == 'docling':
                    # Check if DoclingParser supports this file type
                    if document_parser.is_supported_file(actual_file_path):
                        # Get the selected format
                        use_markdown = st.session_state.get('docling_format', 'markdown') == 'markdown'
                        # Show parsing method indication
                        messenger.parsing_method('docling', file_basename)
                        # DoclingParser now returns a dictionary in the same format as DocumentParser
                        parsed_doc = document_parser.parse_document(actual_file_path, use_markdown=use_markdown)
                        # Fix filename to remove temp_parsing_ prefix
                        if parsed_doc and isinstance(parsed_doc, dict):
                            parsed_doc['file_name'] = file_basename
                        parsed_documents.append(parsed_doc)
                    else:
                        # For unsupported files, fall back to fast parsing
                        file_extension = os.path.splitext(actual_file_path)[1].lower()
                        st.warning(f"⚠️ Docling parsing doesn't support {file_extension} files. Using Fast parsing for {file_basename}")
                        # Use the same format as selected for fast parsing
                        use_markdown = st.session_state.get('fast_format', 'markdown') == 'markdown'
                        # Show parsing method indication
                        messenger.parsing_method('fast', file_basename)
                        parsed_doc = DocumentParser().parse_document(actual_file_path, use_markdown=use_markdown)
                        # Fix filename to remove temp_parsing_ prefix
                        if parsed_doc and isinstance(parsed_doc, dict):
                            parsed_doc['file_name'] = file_basename
                        parsed_documents.append(parsed_doc)
                else:
                    # Get the selected format for fast parsing
                    use_markdown = st.session_state.get('fast_format', 'markdown') == 'markdown'
                    # Show parsing method indication
                    messenger.parsing_method('fast', file_basename)
                    # DocumentParser returns document dictionary
                    parsed_doc = document_parser.parse_document(actual_file_path, use_markdown=use_markdown)
                    # Fix filename to remove temp_parsing_ prefix
                    if parsed_doc and isinstance(parsed_doc, dict):
                        parsed_doc['file_name'] = file_basename
                    parsed_documents.append(parsed_doc)
                
                # Update progress for parsing (40-60%)
                progress_bar.progress(40 + int(20 * (i+1) / total_files))
                logger.info(f"Document parsing: Successfully parsed file {i+1}/{total_files}: {file_basename}")
                
            except Exception as e:
                logger.error(f"Document parsing: Failed to parse file {i+1}/{total_files}: {file_ref} - {str(e)}")
                st.warning(f"⚠️ Could not parse {file_ref}: {str(e)}")
            finally:
                # Always clean up temporary file if created
                if temp_path_created and os.path.exists(temp_path_created):
                    try:
                        os.unlink(temp_path_created)
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up temp file {temp_path_created}: {cleanup_error}")
        
        if not parsed_documents:
            progress_bar.progress(0)
            status_text.text("❌ No documents could be parsed")
            st.error("❌ No documents could be parsed")
            return
        
        messenger.complete_operation(True, f"{len(parsed_documents)} documents parsed successfully")
        progress_bar.progress(60)
        
        # Step 3: Extract data
        # Dynamic extraction model display
        extraction_display_name = "Claude" if extraction_model and 'claude' in extraction_model.lower() else "OpenAI GPT-4.1"
        azure_text = " (Azure)" if api_config.get('use_azure', False) else ""
        
        # Check if this is multi-type document extraction
        logger.info(f"Extraction decision - Session extraction_type: {st.session_state.extraction_type}")
        
        if st.session_state.extraction_type == "multi_type":
            logger.info(f"Using Case 1 multi-type extraction for {len(parsed_documents)} documents")
            if not CASE1_AVAILABLE:
                progress_bar.progress(0)
                status_text.text("❌ Multi-type extraction not available")
                st.error("❌ Multi-type document extraction is not available. Please ensure case1_classifier.py is present.")
                return
            status_text.text(f"🔍 Multi-type document extraction: Classifying and routing {len(parsed_documents)} documents using {extraction_display_name}{azure_text}...")
            progress_bar.progress(70)
            
            try:
                # Use Case 1 extractor for multi-type documents
                case1_extractor = Case1Extractor(ai_client=extractor.client)
                extraction_description = st.session_state.text_description or st.session_state.description
                
                # Debug: Log input document count
                logger.info(f"Multi-type extraction: Processing {len(parsed_documents)} input documents")
                
                # Execute Case 1 multi-type extraction
                messenger.start_operation("Multi-type Extraction", f"Processing {len(parsed_documents)} documents")
                
                results = case1_extractor.extract_from_documents(
                    documents=parsed_documents,
                    extraction_description=extraction_description,
                    use_case_name=st.session_state.use_case
                )
                
                messenger.complete_operation(True, f"Extracted {len(results)} records")
                
                # Debug: Log classification results
                if 'classified_documents' in results:
                    total_classified = sum(results['classified_documents'].values())
                    logger.info(f"Multi-type extraction: Classified {total_classified} documents into {len(results['classified_documents'])} types")
                    for doc_type, count in results['classified_documents'].items():
                        logger.info(f"  - {doc_type}: {count} documents")
                
                # Convert Case 1 results to expected format
                if 'results' in results:
                    # Flatten results from different document types
                    flattened_results = []
                    for doc_type, doc_info in results['results'].items():
                        if 'results' in doc_info and isinstance(doc_info['results'], list):
                            # Add document type info to each result
                            for result in doc_info['results']:
                                if isinstance(result, dict):
                                    result['document_type'] = doc_type
                                    result['extractor_path'] = doc_info.get('extractor_path', '')
                                    flattened_results.append(result)
                    
                    # Debug: Log flattened results count
                    logger.info(f"Multi-type extraction: Flattened {len(flattened_results)} results from {len(results['results'])} document types")
                    
                    if flattened_results:
                        results = flattened_results
                    else:
                        results = [{'error': 'No valid results from multi-type extraction'}]
                else:
                    results = [{'error': results.get('error', 'Multi-type extraction failed')}]
                    
            except Exception as case1_error:
                progress_bar.progress(0)
                status_text.text("❌ Multi-type extraction failed")
                st.error(f"❌ Multi-type extraction failed: {str(case1_error)}")
                return
        
        elif st.session_state.extraction_type == "multi_type_with_relationships":
            logger.info(f"Using Case 2 hierarchical extraction for {len(parsed_documents)} documents")
            logger.info(f"Case 2 decision - Session extraction_type: {st.session_state.extraction_type}")
            
            # Check if Case 2 is available
            try:
                from extraction.hierarchical.case2_main import Case2Orchestrator
                CASE2_AVAILABLE = True
            except ImportError as e:
                logger.error(f"Case 2 import failed: {e}")
                CASE2_AVAILABLE = False
            
            if not CASE2_AVAILABLE:
                progress_bar.progress(0)
                status_text.text("❌ Hierarchical extraction not available")
                st.error("❌ Hierarchical document extraction is not available.")
                st.error("**Missing dependency**: This appears to be a missing `pydantic` package.")
                st.info("**Solution**: Please install required dependencies with: `pip install -r requirements.txt`")
                return
            
            # Check if this is actually a Case 2 use case
            use_case_path = f"templates/{st.session_state.use_case}"
            config_path = os.path.join(use_case_path, "config.json")
            
            fallback_to_single_type = False
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    extraction_config = config_data.get('extraction_config', {})
                    config_extraction_type = extraction_config.get('extraction_type', 'single_type')
                    
                    if config_extraction_type != 'multi_type_with_relationships':
                        logger.warning(f"Use case {st.session_state.use_case} is not a Case 2 use case (config type: {config_extraction_type}), falling back to single-type extraction")
                        fallback_to_single_type = True
                    
                    # Additional check: verify that Case 2 structure exists
                    elif 'extraction_strategy' not in extraction_config:
                        logger.warning(f"Use case {st.session_state.use_case} has Case 2 type but missing extraction_strategy field, falling back to single-type extraction")
                        fallback_to_single_type = True
            
            if fallback_to_single_type:
                # Fall back to single-type extraction
                logger.info(f"Using single-type extraction for {len(parsed_documents)} documents")
                logger.info(f"Single-type decision - Session extraction_type: {st.session_state.extraction_type}")
                status_text.text(f"🧠 Extracting data from {len(parsed_documents)} documents using {extraction_display_name}{azure_text}...")
                progress_bar.progress(70)
                
                try:
                    # Execute single-type extraction
                    messenger.start_operation("Single-type Extraction", f"Processing {len(parsed_documents)} documents")
                    
                    results = extractor.extract_batch(
                        documents=parsed_documents,
                        extraction_prompt=extraction_prompt,
                        model_class=pydantic_model_class,
                        additional_instructions=build_additional_instructions()
                    )
                    
                    messenger.complete_operation(True, f"Extracted {len(results)} records")
                except Exception as extraction_error:
                    # Clean up temporary files even if fallback extraction fails
                    _cleanup_temp_files()
                    
                    progress_bar.progress(0)
                    status_text.text("❌ Data extraction failed")
                    st.error(f"❌ Extraction failed: {str(extraction_error)}")
                    return
                
                # Debug information
                if results:
                    logger.info(f"Single-type extraction completed: {len(results)} results")
                else:
                    logger.warning("Single-type extraction returned no results")
                
                progress_bar.progress(100)
                status_text.text("✅ Data extraction completed")
                
                # Clean up temporary files after fallback single-type extraction
                _cleanup_temp_files()
                
                # Don't return here - let the results display section handle the results
                
            else:
                # Proceed with Case 2 extraction
                status_text.text(f"🔗 Hierarchical extraction: Processing {len(parsed_documents)} documents using {extraction_display_name}{azure_text}...")
                progress_bar.progress(70)
                
                try:
                    # Import and use Case 2 AI adapter
                    from extraction.hierarchical.case2_ai_adapter import Case2AIAdapter
                    
                    # Create adapter for Case 2 compatibility
                    ai_adapter = Case2AIAdapter(extractor)
                    
                    # Use Case 2 orchestrator for hierarchical extraction
                    case2_orchestrator = Case2Orchestrator(ai_client=ai_adapter)
                    extraction_description = st.session_state.text_description or st.session_state.description
                    use_case_name = st.session_state.use_case
                    
                    # Check if this is a new use case or loading existing one
                    if st.session_state.get('use_case_loaded', False):
                        # Load existing use case
                        use_case_path = f"templates/{use_case_name}"
                        logger.info(f"Loading existing use case: {use_case_path}")
                        
                        # Check if the existing use case is compatible with Case 2
                        config_path = os.path.join(use_case_path, "config.json")
                        if os.path.exists(config_path):
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                            
                            extraction_config = config_data.get('extraction_config', {})
                            has_extraction_strategy = 'extraction_strategy' in extraction_config and extraction_config['extraction_strategy'] is not None
                            
                            if has_extraction_strategy:
                                # This is a proper Case 2 use case
                                logger.info(f"Loading existing Case 2 use case: {use_case_path}")
                                
                                # Execute Case 2 hierarchical extraction
                                messenger.start_operation("Hierarchical Extraction", f"Processing {len(parsed_documents)} documents")
                                
                                result = case2_orchestrator.extract_documents(parsed_documents, use_case_path)
                                
                                messenger.complete_operation(True, f"Extracted {len(result.consolidated_results)} consolidated records")
                            else:
                                # This is a single-type use case, convert it to Case 2
                                logger.info(f"Converting single-type use case to Case 2: {use_case_name}")
                                
                                # Get the text description from the existing config
                                text_description = extraction_config.get('text_description', '')
                                if not text_description:
                                    # Fallback to session state
                                    text_description = st.session_state.text_description or st.session_state.description
                                
                                if not text_description:
                                    raise ValueError("No text description available for Case 2 conversion")
                                
                                # Extract original model field names for consistency
                                original_model_info = _extract_original_model_info(use_case_path)
                                
                                # Convert the use case to Case 2
                                create_result = case2_orchestrator.create_new_use_case(
                                    description=text_description,
                                    use_case_name=use_case_name,
                                    use_case_path=use_case_path,
                                    original_model_info=original_model_info
                                )
                                
                                if not create_result['success']:
                                    raise ValueError(f"Failed to convert use case to Case 2: {create_result['error']}")
                                
                                # Extract documents with the converted use case
                                messenger.start_operation("Hierarchical Extraction", f"Processing {len(parsed_documents)} documents")
                                
                                result = case2_orchestrator.extract_documents(parsed_documents, use_case_path)
                                
                                messenger.complete_operation(True, f"Extracted {len(result.consolidated_results)} consolidated records")
                        else:
                            # No config file, treat as new use case
                            logger.info(f"No config found, creating new Case 2 use case: {use_case_name}")
                            create_result = case2_orchestrator.create_new_use_case(
                                description=extraction_description,
                                use_case_name=use_case_name,
                                use_case_path=use_case_path
                            )
                            
                            if not create_result['success']:
                                raise ValueError(f"Failed to create use case: {create_result['error']}")
                            
                            # Extract documents
                            messenger.start_operation("Hierarchical Extraction", f"Processing {len(parsed_documents)} documents")
                            
                            result = case2_orchestrator.extract_documents(parsed_documents, use_case_path)
                            
                            messenger.complete_operation(True, f"Extracted {len(result.consolidated_results)} consolidated records")
                        
                    else:
                        # Create new use case
                        use_case_path = f"templates/{use_case_name}"
                        logger.info(f"Creating new Case 2 use case: {use_case_name}")
                        
                        # Create use case first
                        create_result = case2_orchestrator.create_new_use_case(
                            description=extraction_description,
                            use_case_name=use_case_name,
                            use_case_path=use_case_path
                        )
                        
                        if not create_result['success']:
                            raise ValueError(f"Failed to create use case: {create_result['error']}")
                        
                            # Extract documents
                            messenger.start_operation("Hierarchical Extraction", f"Processing {len(parsed_documents)} documents")
                            
                            result = case2_orchestrator.extract_documents(parsed_documents, use_case_path)
                            
                            messenger.complete_operation(True, f"Extracted {len(result.consolidated_results)} consolidated records")
                    
                    # Convert Case 2 result to expected format
                    results = {
                        'extraction_results': result.consolidated_results,
                        'stage_results': result.stage_results,
                        'relationships': result.relationships,
                        'processing_metadata': result.processing_metadata,
                        'extraction_type': 'hierarchical'
                    }
                    
                    # Display extraction results summary
                    messenger.results_summary(results, 'hierarchical')
                    
                    progress_bar.progress(90)
                    status_text.text("💾 Saving hierarchical extraction results...")
                    
                    # Store results in session state
                    st.session_state.extraction_results = results
                    st.session_state.extraction_type = 'multi_type_with_relationships'
                    
                    # Save results to use-case folder 
                    if results and results.get('extraction_results'):
                        try:
                            clean_data = []
                            for record in results['extraction_results']:
                                # Remove internal metadata for saving
                                clean_record = {k: v for k, v in record.items() if not k.startswith('_')}
                                clean_data.append(clean_record)
                            
                            # Save to Excel
                            import pandas as pd
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_file = f"templates/{st.session_state.use_case}/extraction_results_{timestamp}.xlsx"
                            
                            df = pd.DataFrame(clean_data)
                            df.to_excel(output_file, index=False)
                            logger.info(f"Saved {len(clean_data)} hierarchical records to {output_file}")
                            
                        except Exception as e:
                            logger.warning(f"Could not save hierarchical results to file: {e}")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Hierarchical extraction completed")
                    
                    # Clean up temporary files after hierarchical extraction
                    _cleanup_temp_files()
                    
                except Exception as case2_error:
                    logger.error(f"Error in hierarchical extraction: {case2_error}")
                    progress_bar.progress(0)
                    status_text.text("❌ Hierarchical extraction failed")
                    st.error(f"❌ Hierarchical extraction failed: {str(case2_error)}")
                    return
        else:
            # Standard single-type extraction
            logger.info(f"Using single-type extraction for {len(parsed_documents)} documents")
            logger.info(f"Single-type decision - Session extraction_type: {st.session_state.extraction_type}")
            status_text.text(f"🧠 Extracting data from {len(parsed_documents)} documents using {extraction_display_name}{azure_text}...")
            progress_bar.progress(70)
            
            try:
                # Execute single-type extraction
                messenger.start_operation("Single-type Extraction", f"Processing {len(parsed_documents)} documents")
                
                results = extractor.extract_batch(
                    documents=parsed_documents,
                    extraction_prompt=extraction_prompt,
                    model_class=pydantic_model_class,
                    additional_instructions=build_additional_instructions()
                )
                
                messenger.complete_operation(True, f"Extracted {len(results)} records")
            except Exception as extraction_error:
                progress_bar.progress(0)
                status_text.text("❌ Data extraction failed")
                st.error(f"❌ Extraction failed: {str(extraction_error)}")
            
            # Debug information
            with st.expander("🔍 Debug Information", expanded=True):
                st.write("**Parsed Documents Info:**")
                st.write(f"Number of documents: {len(parsed_documents)}")
                if parsed_documents:
                    first_doc = parsed_documents[0]
                    st.write(f"First document type: {type(first_doc)}")
                    if isinstance(first_doc, dict):
                        st.write(f"First document keys: {list(first_doc.keys())}")
                        st.write(f"Text content preview: {first_doc.get('text_content', 'No text_content')[:200]}...")
                        st.write(f"File name: {first_doc.get('file_name', 'Unknown')}")
                        st.write(f"Parsing method: {first_doc.get('parsing_method', 'Unknown')}")
                    else:
                        st.write(f"First document preview: {str(first_doc)[:200]}...")
                else:
                    st.write("No documents parsed")
                st.write(f"Parsing method used: {parsing_method}")
                st.write(f"DoclingParser available: {DOCLING_AVAILABLE}")
        
        progress_bar.progress(90)
        status_text.text("✅ Data extraction completed")
        
        # Save results 
        status_text.text("💾 Saving extraction results...")
        st.session_state.extraction_results = results
        
        # Save results to use-case folder
        if results:
            try:
                clean_data = []
                for result in results:
                    clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
                    for key, value in clean_result.items():
                        if hasattr(value, 'value'):
                            clean_result[key] = value.value
                        elif isinstance(value, list):
                            if value and hasattr(value[0], 'value'):
                                clean_result[key] = '; '.join([item.value for item in value])
                            else:
                                clean_result[key] = '; '.join([str(item) for item in value])
                    clean_data.append(clean_result)
                
                df = pd.DataFrame(clean_data)
                results_filename = f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                results_path = get_use_case_path(use_case_name, results_filename)
                df.to_excel(results_path, index=False, engine='openpyxl')
                
                st.info(f"📁 Results also saved to: {results_path}")
            except Exception as save_error:
                st.warning(f"⚠️ Could not save results to use-case folder: {str(save_error)}")
        
        # Clean up temporary files
        _cleanup_temp_files()
        
        # Save extraction context to config.json after successful extraction
        save_extraction_context_to_config()
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("🎉 Extraction completed successfully!")
        
        # Clear progress bar after a short delay
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Extraction completed! Processed {len(results)} documents")
        st.rerun()
        
    except Exception as e:
        # Clean up temporary files even if extraction fails
        _cleanup_temp_files()
        
        st.error(f"❌ Extraction failed: {str(e)}")
        
        # Show debug information
        with st.expander("🔍 Debug Information"):
            st.text("Error details:")
            st.text(str(e))
            if 'config' in locals():
                st.text("Configuration:")
                st.json(config)
            
            st.text("Troubleshooting tips:")
            st.text("1. Check that your API keys are correctly set in .env file")
            st.text("2. Ensure field names contain only letters, numbers, and spaces")
            st.text("3. Keep field descriptions concise and clear")
            st.text("4. For enum fields, make sure categories are provided")

def _cleanup_temp_files():
    """Clean up temporary files created during document upload"""
    import os
    import glob
    
    try:
        # Clean up temp files from uploaded documents
        if hasattr(st.session_state, 'selected_files') and st.session_state.selected_files:
            for file_path in st.session_state.selected_files:
                if file_path and file_path.startswith("temp_") and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        logger.info(f"Cleaned up temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not clean up temp file {file_path}: {e}")
        
        # Also clean up any orphaned temp_ files in the project root
        temp_files = glob.glob("temp_*")
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up orphaned temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not clean up orphaned temp file {temp_file}: {e}")
                
        # Clear the selected files from session state
        st.session_state.selected_files = []
        
        # Also clear uploaded file data and reset processing flag to prevent recreation
        if hasattr(st.session_state, 'uploaded_file_data'):
            st.session_state.uploaded_file_data = []
        st.session_state.files_processed = False
        
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

def display_results():
    """Display extraction results in a well-formatted box"""
    results = st.session_state.extraction_results
    
    if not results:
        st.warning("No results to display")
        return
    
    # Create the results container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="results-header">📊 Extraction Results</div>', unsafe_allow_html=True)
    
    # Results summary
    st.markdown('<div class="result-summary">', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Check extraction type first to handle different result formats
    is_hierarchical = st.session_state.get('extraction_type', 'single_type') == 'multi_type_with_relationships'
    
    with col1:
        if is_hierarchical and isinstance(results, dict):
            # Case 2: results is a dict with processing_metadata
            docs_processed = results.get('processing_metadata', {}).get('total_documents', 0)
            st.metric("📄 Documents Processed", docs_processed)
        else:
            # Case 1 or single-type: results is a list
            st.metric("📄 Documents Processed", len(results) if isinstance(results, list) else 0)
    
    with col2:
        if is_hierarchical and isinstance(results, dict):
            # Case 2: Count consolidated results
            consolidated_count = len(results.get('extraction_results', []))
            st.metric("✅ Consolidated Records", consolidated_count)
        else:
            # Case 1 or single-type: Count successful extractions
            successful_extractions = len([r for r in results if not r.get('_document_metadata', {}).get('extraction_error')]) if isinstance(results, list) else 0
            st.metric("✅ Successful", successful_extractions)
    
    with col3:
        if is_hierarchical and isinstance(results, dict):
            # Case 2: Show stages executed
            stages_executed = results.get('processing_metadata', {}).get('stages_executed', 0)
            st.metric("🔗 Stages Executed", stages_executed)
        else:
            # Case 1 or single-type: Show failed extractions
            successful_extractions = len([r for r in results if not r.get('_document_metadata', {}).get('extraction_error')]) if isinstance(results, list) else 0
            docs_count = len(results) if isinstance(results, list) else 0
            failed_extractions = docs_count - successful_extractions
            st.metric("❌ Failed", failed_extractions)
    
    with col4:
        if is_hierarchical and isinstance(results, dict):
            # Case 2: Show total extracted records across all stages
            total_records = results.get('processing_metadata', {}).get('total_extracted_records', 0)
            st.metric("📊 Total Records", total_records)
        else:
            # Case 1 or single-type: Show fields extracted
            if results and isinstance(results, list) and len(results) > 0:
                first_result = {k: v for k, v in results[0].items() if not k.startswith('_')}
                st.metric("📊 Fields Extracted", len(first_result))
            else:
                st.metric("📊 Fields Extracted", 0)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if this is multi-type extraction results based on session state
    is_multi_type = st.session_state.get('extraction_type', 'single_type') == 'multi_type'
    is_hierarchical = st.session_state.get('extraction_type', 'single_type') == 'multi_type_with_relationships'
    
    if is_hierarchical:
        # Case 2 hierarchical results display
        display_hierarchical_results(results)
    elif is_multi_type:
        # Case 1 multi-type results display
        display_multi_type_results(results)
    else:
        # Single-type results display (original format)
        display_single_type_results(results)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_hierarchical_results(results):
    """Display Case 2 hierarchical extraction results"""
    st.markdown("### 🔗 Hierarchical Extraction Results")
    
    # Handle case where results might be a list instead of dict
    if isinstance(results, list):
        st.error("❌ Invalid results format: Expected hierarchical results but got list format.")
        st.info("💡 This usually happens when loading a use case that was created with a different extraction type.")
        return
    
    # Extract results data
    extraction_results = results.get('extraction_results', [])
    stage_results = results.get('stage_results', {})
    relationships = results.get('relationships', {})
    processing_metadata = results.get('processing_metadata', {})
    
    # Display processing summary
    st.markdown("#### 📊 Processing Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", processing_metadata.get('total_documents', 0))
    
    with col2:
        st.metric("🔧 Stages Executed", processing_metadata.get('stages_executed', 0))
    
    with col3:
        st.metric("📋 Total Records", processing_metadata.get('total_extracted_records', 0))
    
    with col4:
        st.metric("🎯 Consolidated Records", len(extraction_results))
    
    # Display stage-by-stage results
    if stage_results:
        st.markdown("#### 🔄 Extraction Stages")
        
        for stage_name, stage_data in stage_results.items():
            with st.expander(f"📋 Stage: {stage_name} ({len(stage_data)} records)"):
                if stage_data:
                    # Create DataFrame for this stage
                    stage_df = pd.DataFrame(stage_data)
                    st.dataframe(stage_df, use_container_width=True)
                else:
                    st.info("No data extracted in this stage")
    
    # Display relationships
    if relationships:
        st.markdown("#### 🔗 Data Flow Between Stages")
        
        for stage_name, rel_data in relationships.items():
            if rel_data:
                st.markdown(f"**{stage_name}** →")
                for target_stage, rel_info in rel_data.items():
                    st.markdown(f"  - **{target_stage}**: via `{rel_info['key_field']}` ({rel_info['relationship_type']})")
    
    # Display consolidated results
    if extraction_results:
        st.markdown("#### 🎯 Consolidated Results")
        
        # Create DataFrame for consolidated results
        consolidated_df = pd.DataFrame(extraction_results)
        
        # Display the consolidated data
        st.dataframe(
            consolidated_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = consolidated_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Consolidated Results as CSV",
            data=csv,
            file_name=f"hierarchical_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No consolidated results available")

def display_multi_type_results(results):
    """Display multi-type extraction results grouped by document type"""
    st.markdown("### 📋 Extracted Data by Document Type")
    
    # Handle case where results might be a dict instead of list
    if isinstance(results, dict):
        st.error("❌ Invalid results format: Expected multi-type results but got dictionary format.")
        st.info("💡 This usually happens when loading a use case that was created with a different extraction type.")
        return
    
    # Group results by document type
    grouped_results = {}
    for result in results:
        doc_type = result.get('document_type', 'unknown')
        if doc_type not in grouped_results:
            grouped_results[doc_type] = []
        grouped_results[doc_type].append(result)
    
    # Display each document type in separate sections
    for doc_type, doc_results in grouped_results.items():
        st.markdown(f"#### 📄 {doc_type.title()} Documents ({len(doc_results)} documents)")
        
        # Create clean DataFrame for this document type (only relevant fields)
        clean_data = []
        for result in doc_results:
            clean_result = {}
            
            # Process extracted fields only (exclude metadata and document_type)
            for key, value in result.items():
                if not key.startswith('_') and key != 'document_type' and key != 'extractor_path':
                    if hasattr(value, 'value'):  # Enum object
                        clean_result[key] = value.value
                    elif isinstance(value, list):
                        if value and hasattr(value[0], 'value'):  # List of enums
                            clean_result[key] = '; '.join([item.value for item in value])
                        else:  # List of strings
                            clean_result[key] = '; '.join([str(item) for item in value])
                    else:
                        clean_result[key] = str(value) if value is not None else 'n/a'
            
            # Add document identifier
            clean_result['Document'] = result.get('_document_metadata', {}).get('file_name', f'Document {len(clean_data) + 1}')
            clean_data.append(clean_result)
        
        if clean_data:
            df = pd.DataFrame(clean_data)
            
            # Reorder columns to put Document first
            cols = ['Document'] + [col for col in df.columns if col != 'Document']
            df = df[cols]
            
            # Display the data table
            st.dataframe(
                df, 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No data extracted from {doc_type} documents.")
    
    # Download section for multi-type
    st.markdown('<div class="download-container">', unsafe_allow_html=True)
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CSV download - create combined data
        combined_data = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                if not key.startswith('_'):
                    if hasattr(value, 'value'):
                        clean_result[key] = value.value
                    elif isinstance(value, list):
                        if value and hasattr(value[0], 'value'):
                            clean_result[key] = '; '.join([item.value for item in value])
                        else:
                            clean_result[key] = '; '.join([str(item) for item in value])
                    else:
                        clean_result[key] = str(value) if value is not None else 'n/a'
            combined_data.append(clean_result)
        
        df_combined = pd.DataFrame(combined_data)
        csv_data = df_combined.to_csv(index=False)
        st.download_button(
            "📥 CSV Format",
            data=csv_data,
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_multi_type_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        excel_buffer = BytesIO()
        df_combined.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        st.download_button(
            "📊 Excel Format",
            data=excel_buffer.getvalue(),
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_multi_type_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # JSON download
        json_data = json.dumps([{k: v for k, v in result.items() if not k.startswith('_')} for result in results], indent=2, default=str)
        st.download_button(
            "📋 JSON Format",
            data=json_data,
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_multi_type_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col4:
        # Clear results
        if st.button("🔄 New Extraction", use_container_width=True):
            st.session_state.extraction_results = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_single_type_results(results):
    """Display single-type extraction results in the original format"""
    # Handle case where results might be a dict instead of list
    if isinstance(results, dict):
        st.error("❌ Invalid results format: Expected single-type results but got dictionary format.")
        st.info("💡 This usually happens when loading a use case that was created with a different extraction type.")
        return
    
    # Create clean DataFrame for display (only extracted fields)
    clean_data = []
    for i, result in enumerate(results):
        clean_result = {}
        
        # Process extracted fields only
        for key, value in result.items():
            if not key.startswith('_'):
                if hasattr(value, 'value'):  # Enum object
                    clean_result[key] = value.value
                elif isinstance(value, list):
                    if value and hasattr(value[0], 'value'):  # List of enums
                        clean_result[key] = '; '.join([item.value for item in value])
                    else:  # List of strings
                        clean_result[key] = '; '.join([str(item) for item in value])
                else:
                    clean_result[key] = str(value) if value is not None else 'n/a'
        
        clean_data.append(clean_result)
    
    df = pd.DataFrame(clean_data)
    
    # Display the data table with enhanced formatting
    st.markdown("### 📋 Extracted Data")
    
    # Show extracted data table
    st.dataframe(
        df, 
        use_container_width=True,
        hide_index=True
    )
    
    if len(results) > 10:
        st.info(f"Showing all {len(results)} documents processed.")
    
    # Download section
    st.markdown('<div class="download-container">', unsafe_allow_html=True)
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CSV download
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 CSV Format",
            data=csv_data,
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        st.download_button(
            "📊 Excel Format",
            data=excel_buffer.getvalue(),
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # JSON download
        json_data = json.dumps([{k: v for k, v in result.items() if not k.startswith('_')} for result in results], indent=2, default=str)
        st.download_button(
            "📋 JSON Format",
            data=json_data,
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col4:
        # Clear results
        if st.button("🔄 New Extraction", use_container_width=True):
            st.session_state.extraction_results = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main UI function"""
    initialize_session_state()
    
    # Ensure templates folder exists at startup
    ensure_use_cases_folder()
    
    # Header
    st.markdown('<h1 class="main-header">📑Knowledge Extraction Agent</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    col1, col2 = st.columns(2)
    with col1:
        config_style = "primary" if st.session_state.current_tab == "Configuration" else "secondary"
        if st.button("⚙️ Configuration", type=config_style, use_container_width=True):
            st.session_state.current_tab = "Configuration"
            st.rerun()
    
    with col2:
        extraction_style = "primary" if st.session_state.current_tab == "Extraction" else "secondary"
        if st.button("🎯 Extraction", type=extraction_style, use_container_width=True):
            load_extraction_context_from_current_config()  # Load context before switching
            st.session_state.current_tab = "Extraction"
            st.rerun()
    
    st.markdown("---")
    
    # Session state-based tab navigation
    if st.session_state.current_tab == "Configuration":
        configuration_section()
    elif st.session_state.current_tab == "Extraction":
        extraction_section()

if __name__ == "__main__":
    # Add required import for Excel download
    from io import BytesIO
    main()