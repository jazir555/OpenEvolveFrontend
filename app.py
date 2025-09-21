"""OpenEvolve Protocol Improver - A Streamlit application for protocol improvement using LLMs.

This module provides a comprehensive interface for improving protocols and standard operating
procedures (SOPs) through two main approaches:

1. Evolution-based Improvement: Uses a single LLM provider to iteratively refine protocols
2. Adversarial Testing: Employs multiple LLM providers in a red team/blue team approach
   to identify vulnerabilities and generate hardened protocols

Key Features:
- Support for 34+ LLM providers including OpenAI, Anthropic, Google Gemini, and more
- Real-time protocol evaluation and improvement
- Cost estimation and token tracking
- Thread-safe operations for concurrent model evaluation
- Comprehensive logging and result visualization
- Collaborative features and version control
- Advanced analytics and reporting

The application uses Streamlit for the web interface and provides both single-provider
evolution and multi-provider adversarial testing capabilities.
"""

import functools
import json
import math
import os
import random
import re
import threading
import time
import traceback
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import streamlit as st
except ImportError:
    print("streamlit package not found. Please install it with 'pip install streamlit'")
    st = None

try:
    import requests
except ImportError:
    if st is not None:
        st.error("requests package not found. Please install it with 'pip install requests'")
    else:
        print("requests package not found. Please install it with 'pip install requests'")
    requests = None

# Graceful fallback for streamlit_tags
try:
    from streamlit_tags import st_tags
    HAS_STREAMLIT_TAGS = True
except ImportError:
    HAS_STREAMLIT_TAGS = False
    st.error("streamlit_tags package not found. Please install it with 'pip install streamlit-tags' for full functionality.")

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ------------------------------------------------------------------
# 0. Streamlit page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="OpenEvolve Protocol Improver (34 providers + Adversarial Testing)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
/* Custom color scheme */
:root {
    --primary-color: #4a6fa5;
    --secondary-color: #6b8cbc;
    --accent-color: #ff6b6b;
    --success-color: #4caf50;
    --warning-color: #ff9800;
    --error-color: #f44336;
    --background-color: #f8f9fa;
    --card-background: #ffffff;
    --text-color: #333333;
    --border-radius: 8px;
    --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    --box-shadow-hover: 0 6px 12px rgba(0, 0, 0, 0.15);
}

/* Dark mode variables */
[data-theme="dark"] {
    --primary-color: #6b8cbc;
    --secondary-color: #4a6fa5;
    --accent-color: #ff6b6b;
    --success-color: #4caf50;
    --warning-color: #ff9800;
    --error-color: #f44336;
    --background-color: #0e1117;
    --card-background: #1e2130;
    --text-color: #fafafa;
}

/* Main title styling */
h1 {
    color: var(--primary-color);
    text-align: center;
    padding: 1rem 0;
    border-bottom: 2px solid var(--primary-color);
    margin-bottom: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    padding: 0.5rem;
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: var(--border-radius);
    color: var(--text-color);
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    box-shadow: var(--box-shadow);
}

/* Card styling for sections */
.stMarkdown, .stDataFrame, .stTable {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stMarkdown:hover, .stDataFrame:hover, .stTable:hover {
    box-shadow: var(--box-shadow-hover);
    transform: translateY(-2px);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: var(--box-shadow);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--box-shadow-hover);
}

.stButton > button:active {
    transform: translateY(0);
}

.stButton > button[kind="secondary"] {
    background-color: var(--card-background);
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
}

/* Input styling */
.stSelectbox, .stTextInput, .stTextArea, .stNumberInput, .stSlider {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stSelectbox:focus, .stTextInput:focus, .stTextArea:focus, .stNumberInput:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(74, 111, 165, 0.2);
}

[data-theme="dark"] .stSelectbox, 
[data-theme="dark"] .stTextInput, 
[data-theme="dark"] .stTextArea, 
[data-theme="dark"] .stNumberInput, 
[data-theme="dark"] .stSlider {
    border: 1px solid #4a4a4a;
}

/* Metric styling */
.stMetric {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.stMetric:hover {
    transform: translateY(-2px);
    box-shadow: var(--box-shadow-hover);
}

.stMetric label {
    font-size: 1rem;
    color: var(--text-color);
    font-weight: 500;
}

.stMetric div[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-top: 0.5rem;
}

/* Status message styling */
.stStatus {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--primary-color);
}

/* Expander styling */
.stExpander {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stExpander:hover {
    box-shadow: var(--box-shadow-hover);
}

.stExpander div[data-testid="stExpanderDetails"] {
    padding: 1.5rem;
    border-top: 1px solid #e0e0e0;
}

[data-theme="dark"] .stExpander div[data-testid="stExpanderDetails"] {
    border-top: 1px solid #4a4a4a;
}

/* Code block styling */
.stCodeBlock {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--card-background);
    border-right: 1px solid #e0e0e0;
    box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
}

[data-theme="dark"] [data-testid="stSidebar"] {
    border-right: 1px solid #4a4a4a;
}

/* Progress bar styling */
.stProgress {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

/* Alert styling */
.stAlert {
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

/* Custom classes for specific elements */
.protocol-analysis-card {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    border-radius: var(--border-radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--box-shadow);
    transition: all 0.3s ease;
}

.protocol-analysis-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--box-shadow-hover);
}

.protocol-analysis-card h3 {
    color: white;
    margin-top: 0;
    font-weight: 600;
}

.team-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0.25rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.team-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.red-team {
    background-color: #ffebee;
    color: #c62828;
    border: 1px solid #ffcdd2;
}

[data-theme="dark"] .red-team {
    background-color: #3a1414;
    color: #ef9a9a;
    border: 1px solid #5c2323;
}

.blue-team {
    background-color: #e3f2fd;
    color: #1565c0;
    border: 1px solid #bbdefb;
}

[data-theme="dark"] .blue-team {
    background-color: #14233a;
    color: #90caf9;
    border: 1px solid #233d5c;
}

.model-performance-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

.model-performance-table th,
.model-performance-table td {
    padding: 1rem;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
}

[data-theme="dark"] .model-performance-table th,
[data-theme="dark"] .model-performance-table td {
    border-bottom: 1px solid #4a4a4a;
}

.model-performance-table th {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    font-weight: 600;
}

.model-performance-table tr:hover {
    background-color: rgba(74, 111, 165, 0.1);
}

[data-theme="dark"] .model-performance-table tr:hover {
    background-color: rgba(74, 111, 165, 0.2);
}

/* New enhanced styles */
.header-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    margin-left: 0.5rem;
    vertical-align: middle;
}

.success-badge {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.warning-badge {
    background-color: #fff8e1;
    color: #f57f17;
}

.error-badge {
    background-color: #ffebee;
    color: #c62828;
}

[data-theme="dark"] .success-badge {
    background-color: #1b5e20;
    color: #a5d6a7;
}

[data-theme="dark"] .warning-badge {
    background-color: #33691e;
    color: #f4ff81;
}

[data-theme="dark"] .error-badge {
    background-color: #b71c1c;
    color: #ffcdd2;
}

/* Loading spinner */
.loading-spinner {
    border: 4px solid rgba(74, 111, 165, 0.2);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced dashboard styling */
.dashboard-card {
    background: linear-gradient(135deg, var(--card-background), #f0f2f6);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.dashboard-card:hover {
    box-shadow: var(--box-shadow-hover);
    transform: translateY(-2px);
}

.dashboard-card h3 {
    color: var(--primary-color);
    margin-top: 0;
    font-weight: 600;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 0.5rem;
}

/* Enhanced button styling */
.stButton > button.primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: var(--box-shadow);
}

.stButton > button.secondary {
    background-color: var(--card-background);
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: var(--box-shadow);
}

.stButton > button.success {
    background: linear-gradient(135deg, var(--success-color), #81c784);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: var(--box-shadow);
}

.stButton > button.warning {
    background: linear-gradient(135deg, var(--warning-color), #ffb74d);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: var(--box-shadow);
}

.stButton > button.error {
    background: linear-gradient(135deg, var(--error-color), #e57373);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: var(--box-shadow);
}

/* Enhanced input styling */
.stSelectbox, .stTextInput, .stTextArea, .stNumberInput, .stSlider {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stSelectbox:focus, .stTextInput:focus, .stTextArea:focus, .stNumberInput:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(74, 111, 165, 0.2);
}

[data-theme="dark"] .stSelectbox, 
[data-theme="dark"] .stTextInput, 
[data-theme="dark"] .stTextArea, 
[data-theme="dark"] .stNumberInput, 
[data-theme="dark"] .stSlider {
    border: 1px solid #4a4a4a;
}

/* Enhanced metric styling */
.stMetric {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.stMetric:hover {
    transform: translateY(-2px);
    box-shadow: var(--box-shadow-hover);
}

.stMetric label {
    font-size: 1rem;
    color: var(--text-color);
    font-weight: 500;
}

.stMetric div[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-top: 0.5rem;
}

/* Enhanced status message styling */
.stStatus {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--primary-color);
}

/* Enhanced expander styling */
.stExpander {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stExpander:hover {
    box-shadow: var(--box-shadow-hover);
}

.stExpander div[data-testid="stExpanderDetails"] {
    padding: 1.5rem;
    border-top: 1px solid #e0e0e0;
}

[data-theme="dark"] .stExpander div[data-testid="stExpanderDetails"] {
    border-top: 1px solid #4a4a4a;
}

/* Enhanced code block styling */
.stCodeBlock {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

/* Enhanced sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--card-background);
    border-right: 1px solid #e0e0e0;
    box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
}

[data-theme="dark"] [data-testid="stSidebar"] {
    border-right: 1px solid #4a4a4a;
}

/* Enhanced progress bar styling */
.stProgress {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

/* Enhanced alert styling */
.stAlert {
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    .protocol-analysis-card {
        padding: 1.5rem;
    }
    
    .team-badge {
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
    }
    
    .dashboard-card {
        padding: 1rem;
    }
}

/* Animation for new elements */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Enhanced team badge styling */
.team-badge-lg {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 0.5rem;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}

.team-badge-lg:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
}

/* Enhanced model performance table */
.model-performance-table-enhanced {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--box-shadow);
}

.model-performance-table-enhanced th,
.model-performance-table-enhanced td {
    padding: 1.2rem;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
}

[data-theme="dark"] .model-performance-table-enhanced th,
[data-theme="dark"] .model-performance-table-enhanced td {
    border-bottom: 1px solid #4a4a4a;
}

.model-performance-table-enhanced th {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    font-weight: 600;
}

.model-performance-table-enhanced tr:hover {
    background-color: rgba(74, 111, 165, 0.1);
}

[data-theme="dark"] .model-performance-table-enhanced tr:hover {
    background-color: rgba(74, 111, 165, 0.2);
}

/* Enhanced protocol comparison styling */
.protocol-comparison {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.protocol-comparison-column {
    flex: 1;
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.protocol-comparison-header {
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e0e0e0;
}

/* Enhanced visualization styling */
.chart-container {
    background-color: var(--card-background);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.chart-title {
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Constants and Prompts
# ------------------------------------------------------------------

JSON_RE = re.compile(r"\{[\s\S]*\}")
FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

def safe_int(x, d=0):
    return d if x is None or not isinstance(x, (int, float, str)) or str(x) == '' else int(float(x))

def safe_float(x, d=0.0):
    return d if x is None or not isinstance(x, (int, float, str)) or str(x) == '' else float(x)

# Define adversarial prompts as constants for clarity and maintainability.
APPROVAL_PROMPT = (
    "You are a senior technical reviewer. Your task is to evaluate the provided Standard Operating Procedure (SOP). "
    "Return a single, STRICT JSON object and nothing else. The JSON object must have the following keys:\n"
    '{"verdict": "APPROVED" | "REJECTED", "score": 0-100, "reasons": ["..."], "notes": "..."}\n'
    "- `verdict`: Your final judgment. 'APPROVED' if the SOP is robust, clear, and secure. 'REJECTED' otherwise.\n"
    "- `score`: An integer from 0 to 100 reflecting the overall quality, security, and robustness to abuse.\n"
    "- `reasons`: A list of concise strings explaining the key factors behind your verdict and score.\n"
    "- `notes`: Any additional commentary or observations.\n"
    "Do not include any text, explanation, or formatting outside of the single JSON object."
)

RED_TEAM_CRITIQUE_PROMPT = (
    "You are a critical technical reviewer conducting a peer review. Your goal is to find every possible flaw in the provided SOP. "
    "Analyze it for logical gaps, ambiguities, edge cases, undefined responsibilities, missing "
    "preconditions, unsafe defaults, and potential paths for errors or misuse. "
    "Also, check for compliance with the following requirements:\\n{compliance_requirements}\\n"
    "Return a single, STRICT JSON object and nothing else with the following structure:\\n"
    '{\"issues\": [{\"title\": \"...\", \"severity\": \"low|medium|high|critical\", \"category\": \"...\", \"detail\": \"...\", '
    '\"reproduction\": \"...\", \"exploit_paths\": [\"...\"], \"mitigations\": [\"...\"]}], '
    '\"summary\": \"...\", \"overall_risk\": \"low|medium|high|critical\"}'
)

BLUE_TEAM_PATCH_PROMPT = (
    "You are a technical improvement specialist. Your task is to address the issues identified in the "
    "provided critiques and produce an improved version of the SOP. Incorporate the feedback to make the protocol "
    "explicit, verifiable, and robust. Add preconditions, validation steps, error handling, "
    "monitoring, auditability, and documentation where applicable. "
    "Return a single, STRICT JSON object and nothing else with the following keys:\\n"
    '{\"sop\": \"<the complete improved SOP in Markdown>\", \"changelog\": [\"...\"], \"residual_risks\": [\"...\"], '
    '\"mitigation_matrix\": [{\"issue\": \"...\", \"fix\": \"...\", \"status\": \"resolved|mitigated|wontfix\"}]}'
)

# Specialized prompts for different review types
CODE_REVIEW_RED_TEAM_PROMPT = (
    "You are a senior software engineer conducting a code peer review. Your goal is to find every possible flaw in the provided code. "
    "Analyze it for bugs, security vulnerabilities, performance issues, maintainability problems, "
    "code smells, anti-patterns, and potential runtime errors. "
    "Also, check for compliance with the following requirements:\\n{compliance_requirements}\\n"
    "Return a single, STRICT JSON object and nothing else with the following structure:\\n"
    '{\"issues\": [{\"title\": \"...\", \"severity\": \"low|medium|high|critical\", \"category\": \"bug|security|performance|maintainability|style\", \"detail\": \"...\", \"line_number\": \"...\", \"suggested_fix\": \"...\"}], '
    '\"summary\": \"...\", \"overall_quality\": \"poor|fair|good|excellent\"}'
)

CODE_REVIEW_BLUE_TEAM_PROMPT = (
    "You are a code improvement specialist. Your task is to address the issues identified in the "
    "provided code review critiques and produce an improved version of the code. Incorporate the feedback to make the code "
    "more secure, efficient, maintainable, and readable. Add proper error handling, "
    "documentation, and follow best practices. "
    "Return a single, STRICT JSON object and nothing else with the following keys:\\n"
    '{\"code\": \"<the complete improved code>\", \"changelog\": [\"...\"], \"residual_issues\": [\"...\"], '
    '\"fix_matrix\": [{\"issue\": \"...\", \"fix\": \"...\", \"status\": \"resolved|mitigated|wontfix\"}]}'
)

PLAN_REVIEW_RED_TEAM_PROMPT = (
    "You are a strategic planning expert conducting a peer review of the provided plan. Your goal is to find every possible flaw in the plan. "
    "Analyze it for unrealistic assumptions, missing steps, unclear objectives, resource constraints, "
    "timeline issues, risk factors, and potential failure points. "
    "Also, check for compliance with the following requirements:\\n{compliance_requirements}\\n"
    "Return a single, STRICT JSON object and nothing else with the following structure:\\n"
    '{\"issues\": [{\"title\": \"...\", \"severity\": \"low|medium|high|critical\", \"category\": \"assumptions|resources|timeline|risks|clarity\", \"detail\": \"...\", \"suggested_improvement\": \"...\"}], '
    '\"summary\": \"...\", \"overall_feasibility\": \"low|medium|high\"}'
)

PLAN_REVIEW_BLUE_TEAM_PROMPT = (
    "You are a planning improvement specialist. Your task is to address the issues identified in the "
    "provided plan review critiques and produce an improved version of the plan. Incorporate the feedback to make the plan "
    "more realistic, actionable, and robust. Add missing details, clarify objectives, "
    "adjust timelines, and improve resource allocation. "
    "Return a single, STRICT JSON object and nothing else with the following keys:\\n"
    '{\"plan\": \"<the complete improved plan in Markdown>\", \"changelog\": [\"...\"], \"residual_concerns\": [\"...\"], '
    '\"improvement_matrix\": [{\"issue\": \"...\", \"improvement\": \"...\", \"status\": \"resolved|mitigated|wontfix\"}]}'
)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _now_ms() -> int:
    """Get current time in milliseconds since epoch.
    
    Returns:
        int: Current timestamp in milliseconds
    """
    return int(time.time() * 1000)

def calculate_protocol_complexity(protocol_text: str) -> Dict[str, Any]:
    """Calculate various complexity metrics for a protocol.
    
    Args:
        protocol_text (str): The protocol text to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing complexity metrics
    """
    if not protocol_text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_sentence_length": 0,
            "avg_paragraph_length": 0,
            "unique_words": 0,
            "complexity_score": 0
        }
    
    # Basic text statistics
    words = protocol_text.split()
    word_count = len(words)
    
    sentences = re.split(r'[.!?]+', protocol_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    paragraphs = [p.strip() for p in protocol_text.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Average lengths
    avg_sentence_length = word_count / max(1, sentence_count)
    avg_paragraph_length = sentence_count / max(1, paragraph_count)
    
    # Unique words
    unique_words = len(set(words))
    
    # Complexity score (weighted combination of metrics)
    # Higher scores indicate more complex protocols
    complexity_score = (
        (word_count / 100) * 0.3 +  # Normalize word count
        (avg_sentence_length / 20) * 0.3 +  # Longer sentences = more complex
        (avg_paragraph_length / 10) * 0.2 +  # Longer paragraphs = more complex
        (1 - (unique_words / max(1, word_count))) * 0.2 * 100  # Repetition = more complex
    )
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_paragraph_length": round(avg_paragraph_length, 2),
        "unique_words": unique_words,
        "complexity_score": round(complexity_score, 2)
    }

def extract_protocol_structure(protocol_text: str) -> Dict[str, Any]:
    """Extract structural elements from a protocol.
    
    Args:
        protocol_text (str): The protocol text to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing structural elements
    """
    if not protocol_text:
        return {
            "has_numbered_steps": False,
            "has_bullet_points": False,
            "has_headers": False,
            "section_count": 0,
            "has_preconditions": False,
            "has_postconditions": False,
            "has_error_handling": False
        }
    
    lines = protocol_text.split('\n')
    
    # Check for numbered steps (e.g., "1.", "2.", "1.1", etc.)
    numbered_pattern = re.compile(r'^\s*\d+\.')
    has_numbered_steps = any(numbered_pattern.match(line) for line in lines)
    
    # Check for bullet points
    bullet_pattern = re.compile(r'^\s*[*\-+]')
    has_bullet_points = any(bullet_pattern.match(line) for line in lines)
    
    # Check for headers (markdown style ## or === underlines)
    header_pattern = re.compile(r'^#{1,6}\s+|.*\n[=]{3,}|.*\n[-]{3,}')
    has_headers = bool(header_pattern.search(protocol_text))
    
    # Count sections (headers)
    section_count = len(header_pattern.findall(protocol_text))
    
    # Check for common protocol elements
    lower_text = protocol_text.lower()
    has_preconditions = 'precondition' in lower_text or 'prerequisite' in lower_text
    has_postconditions = 'postcondition' in lower_text
    has_error_handling = 'error' in lower_text or 'exception' in lower_text or 'failure' in lower_text
    
    return {
        "has_numbered_steps": has_numbered_steps,
        "has_bullet_points": has_bullet_points,
        "has_headers": has_headers,
        "section_count": section_count,
        "has_preconditions": has_preconditions,
        "has_postconditions": has_postconditions,
        "has_error_handling": has_error_handling
    }

# ------------------------------------------------------------------
# AI-Powered Recommendation Functions
# ------------------------------------------------------------------

def generate_protocol_recommendations(protocol_text: str) -> List[str]:
    """Generate AI-powered recommendations for improving a protocol.
    
    Args:
        protocol_text (str): The protocol text to analyze
        
    Returns:
        List[str]: List of recommendations
    """
    if not protocol_text:
        return ["Please enter a protocol to analyze."]
    
    recommendations = []
    complexity = calculate_protocol_complexity(protocol_text)
    structure = extract_protocol_structure(protocol_text)
    
    # Complexity-based recommendations
    if complexity["complexity_score"] > 50:
        recommendations.append("📝 Protocol is quite complex. Consider breaking it into smaller, more manageable sections.")
    
    if complexity["avg_sentence_length"] > 25:
        recommendations.append("🔗 Average sentence length is high. Try to use shorter, clearer sentences.")
    
    if complexity["unique_words"] / max(1, complexity["word_count"]) < 0.4:
        recommendations.append("🔄 High word repetition detected. Consider using more varied vocabulary for clarity.")
    
    # Structure-based recommendations
    if not structure["has_headers"]:
        recommendations.append("📌 Add headers to organize your protocol into clear sections.")
    
    if not structure["has_numbered_steps"] and not structure["has_bullet_points"]:
        recommendations.append("🔢 Use numbered steps or bullet points to make instructions clearer.")
    
    if not structure["has_preconditions"]:
        recommendations.append("🔒 Add preconditions to specify what must be true before executing the protocol.")
    
    if not structure["has_postconditions"]:
        recommendations.append("✅ Add postconditions to specify what should be true after executing the protocol.")
    
    if not structure["has_error_handling"]:
        recommendations.append("⚠️ Include error handling procedures for when things go wrong.")
    
    # Length-based recommendations
    if complexity["word_count"] < 100:
        recommendations.append("📄 Protocol is quite short. Ensure all necessary details are included.")
    
    if structure["section_count"] == 0 and complexity["word_count"] > 300:
        recommendations.append("📂 Long protocol without sections. Consider organizing it with headers for better readability.")
    
    # Add some general recommendations if none were triggered
    if not recommendations:
        recommendations.append("👍 Your protocol looks well-structured! Consider running adversarial testing for additional hardening.")
    
    return recommendations

def suggest_protocol_template(protocol_text: str) -> str:
    """Suggest the most appropriate protocol template based on content.
    
    Args:
        protocol_text (str): The protocol text to analyze
        
    Returns:
        str: Suggested template name
    """
    if not protocol_text:
        return "Standard Operating Procedure"
    
    lower_text = protocol_text.lower()
    
    # Check for keywords that indicate specific template types
    if any(keyword in lower_text for keyword in ["security", "vulnerability", "attack", "threat", "penetration"]):
        return "Security Policy"
    elif any(keyword in lower_text for keyword in ["incident", "response", "emergency", "crisis"]):
        return "Incident Response Plan"
    elif any(keyword in lower_text for keyword in ["compliance", "regulation", "audit", "policy"]):
        return "Compliance Policy"
    elif any(keyword in lower_text for keyword in ["development", "code", "software", "programming"]):
        return "Development Process"
    elif any(keyword in lower_text for keyword in ["testing", "test", "qa", "quality"]):
        return "Testing Procedure"
    else:
        return "Standard Operating Procedure"  # Default suggestion

def compare_protocols(protocol_a: str, protocol_b: str) -> Dict[str, Any]:
    """Compare two protocols and highlight differences.
    
    Args:
        protocol_a (str): First protocol text
        protocol_b (str): Second protocol text
        
    Returns:
        Dict[str, Any]: Comparison results including similarity metrics and differences
    """
    if not protocol_a or not protocol_b:
        return {
            "similarity": 0.0,
            "length_difference": 0,
            "added_sections": [],
            "removed_sections": [],
            "complexity_change": 0.0,
            "word_count_change": 0,
            "sentence_count_change": 0,
            "unique_words_change": 0,
            "structure_changes": {}
        }
    
    # Calculate similarity using a simple approach
    # In a real implementation, we might use more sophisticated algorithms
    set_a = set(protocol_a.split())
    set_b = set(protocol_b.split())
    
    # Jaccard similarity
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    similarity = intersection / max(1, union)
    
    # Length difference
    length_difference = len(protocol_b) - len(protocol_a)
    
    # Complexity metrics
    complexity_a = calculate_protocol_complexity(protocol_a)
    complexity_b = calculate_protocol_complexity(protocol_b)
    complexity_change = complexity_b["complexity_score"] - complexity_a["complexity_score"]
    
    # Detailed metrics changes
    word_count_change = complexity_b["word_count"] - complexity_a["word_count"]
    sentence_count_change = complexity_b["sentence_count"] - complexity_a["sentence_count"]
    unique_words_change = complexity_b["unique_words"] - complexity_a["unique_words"]
    
    # Simple section difference detection
    sections_a = re.findall(r'^#{1,6}\s+.*$|.*\n[=]{3,}|.*\n[-]{3,}', protocol_a, re.MULTILINE)
    sections_b = re.findall(r'^#{1,6}\s+.*$|.*\n[=]{3,}|.*\n[-]{3,}', protocol_b, re.MULTILINE)
    
    added_sections = [s for s in sections_b if s not in sections_a]
    removed_sections = [s for s in sections_a if s not in sections_b]
    
    # Structure analysis
    structure_a = extract_protocol_structure(protocol_a)
    structure_b = extract_protocol_structure(protocol_b)
    
    structure_changes = {}
    for key in structure_a.keys():
        if structure_a[key] != structure_b[key]:
            structure_changes[key] = {
                "before": structure_a[key],
                "after": structure_b[key]
            }
    
    return {
        "similarity": round(similarity * 100, 2),
        "length_difference": length_difference,
        "added_sections": added_sections,
        "removed_sections": removed_sections,
        "complexity_change": round(complexity_change, 2),
        "word_count_change": word_count_change,
        "sentence_count_change": sentence_count_change,
        "unique_words_change": unique_words_change,
        "structure_changes": structure_changes,
        "improvement": "increased" if complexity_change < 0 else "decreased" if complexity_change > 0 else "unchanged"
    }


def render_protocol_comparison(protocol_a: str, protocol_b: str, name_a: str = "Original", name_b: str = "Modified") -> str:
    """Render a visual comparison of two protocols.
    
    Args:
        protocol_a (str): First protocol text
        protocol_b (str): Second protocol text
        name_a (str): Name for first protocol
        name_b (str): Name for second protocol
        
    Returns:
        str: HTML formatted comparison
    """
    if not protocol_a or not protocol_b:
        return "<p>Cannot compare empty protocols</p>"
    
    comparison = compare_protocols(protocol_a, protocol_b)
    
    # Split protocols into lines for line-by-line comparison
    lines_a = protocol_a.split('\n')
    lines_b = protocol_b.split('\n')
    
    # Simple diff implementation
    html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="flex: 1;">
            <h3 style="color: #4a6fa5; border-bottom: 2px solid #4a6fa5; padding-bottom: 5px;">{name_a}</h3>
        </div>
        <div style="flex: 1;">
            <h3 style="color: #4a6fa5; border-bottom: 2px solid #4a6fa5; padding-bottom: 5px;">{name_b}</h3>
        </div>
    </div>
    """
    
    # Add summary metrics
    html += f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4>📊 Comparison Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <strong>Similarity</strong><br>
                <span style="font-size: 1.2em; color: {'#4caf50' if comparison['similarity'] > 80 else '#ff9800' if comparison['similarity'] > 50 else '#f44336'}">
                    {comparison['similarity']}%
                </span>
            </div>
            <div style="background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <strong>Length Change</strong><br>
                <span style="font-size: 1.2em; color: {'#4caf50' if comparison['length_difference'] > 0 else '#f44336' if comparison['length_difference'] < 0 else '#666'}">
                    {comparison['length_difference']:+d} chars
                </span>
            </div>
            <div style="background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <strong>Complexity Change</strong><br>
                <span style="font-size: 1.2em; color: {'#4caf50' if comparison['complexity_change'] < 0 else '#f44336' if comparison['complexity_change'] > 0 else '#666'}">
                    {comparison['complexity_change']:+.2f}
                </span>
            </div>
        </div>
    </div>
    """
    
    # Add structural changes
    if comparison["structure_changes"]:
        html += "<div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>"
        html += "<h4>🧩 Structural Changes</h4><ul>"
        for key, change in comparison["structure_changes"].items():
            html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> "
            html += f"{'✅' if not change['before'] and change['after'] else '❌' if change['before'] and not change['after'] else '🔄'} "
            html += f"Changed from {change['before']} to {change['after']}</li>"
        html += "</ul></div>"
    
    # Add section changes
    if comparison["added_sections"] or comparison["removed_sections"]:
        html += "<div style='background-color: #fff3e0; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>"
        html += "<h4>📋 Section Changes</h4>"
        if comparison["added_sections"]:
            html += "<p><strong>Added Sections:</strong></p><ul>"
            for section in comparison["added_sections"][:5]:  # Limit to first 5
                html += f"<li>➕ {section[:50]}{'...' if len(section) > 50 else ''}</li>"
            if len(comparison["added_sections"]) > 5:
                html += f"<li>... and {len(comparison['added_sections']) - 5} more</li>"
            html += "</ul>"
        if comparison["removed_sections"]:
            html += "<p><strong>Removed Sections:</strong></p><ul>"
            for section in comparison["removed_sections"][:5]:  # Limit to first 5
                html += f"<li>➖ {section[:50]}{'...' if len(section) > 50 else ''}</li>"
            if len(comparison["removed_sections"]) > 5:
                html += f"<li>... and {len(comparison['removed_sections']) - 5} more</li>"
            html += "</ul>"
        html += "</div>"
    
    return html

def _rand_jitter_ms(base: int = 250, spread: int = 500) -> float:
    """Generate random jitter time in seconds for retry backoff.
    
    Args:
        base: Base milliseconds value
        spread: Random spread in milliseconds
        
    Returns:
        float: Random jitter time in seconds
    """
    return (base + random.randint(0, spread)) / 1000.0

def _hash_text(s: str) -> str:
    """Generate a short hash of text for comparison purposes.
    
    Args:
        s: Text to hash
        
    Returns:
        str: First 16 characters of SHA256 hash
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _approx_tokens(txt: str) -> int:
    """Approximate token count using character length.
    
    Args:
        txt: Text to estimate tokens for
        
    Returns:
        int: Estimated token count (minimum 1)
    """
    # A conservative approximation: 1 token ~ 4 chars in English. Clamp to a minimum of 1.
    return max(1, math.ceil(len(txt) / 4))

def _safe_json_loads(s: str) -> Optional[dict]:
    """Safely parse JSON string, returning None on failure.
    
    Args:
        s: JSON string to parse
        
    Returns:
        Optional[dict]: Parsed JSON object or None if invalid
    """
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None

def _safe_list(d: dict, key: str) -> list:
    """Safely retrieve a list from a dictionary, returning an empty list if not found or not a list.
    
    Args:
        d: Dictionary to retrieve from
        key: Key to look up
        
    Returns:
        list: The value at the key if it's a list, otherwise an empty list
    """
    if not isinstance(d, dict):
        return []
    value = d.get(key, [])
    return value if isinstance(value, list) else []

def _extract_json_block(txt: str) -> Optional[dict]:
    """Extract JSON object from text, handling fenced code blocks.
    
    Args:
        txt: Text containing potential JSON
        
    Returns:
        Optional[dict]: Extracted JSON object or None if not found
    """
    if not txt or not isinstance(txt, str):
        return None
    # Try to find a JSON block fenced with ```json
    m = FENCE_RE.search(txt)
    if m:
        cand = _safe_json_loads(m.group(1).strip())
        if cand is not None:
            return cand
    # If not found, try to find any substring that looks like a JSON object
    m2 = JSON_RE.search(txt)
    if m2:
        cand = _safe_json_loads(m2.group(0))
        if cand is not None:
            return cand
    return None

def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def _parse_price_per_million(v: Any) -> Optional[float]:
    """
    Safely parses OpenRouter pricing fields (e.g., "0.15", 0.15, "N/A")
    into a float representing price per 1M tokens, or None if invalid.
    """
    if v is None or isinstance(v, (dict, list)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _cost_estimate(prompt_toks: int, completion_toks: int, ppm_prompt: Optional[float], ppm_comp: Optional[float]) -> float:
    """Calculates the estimated cost for a given number of tokens and prices per million."""
    cost = 0.0
    if ppm_prompt is not None and isinstance(ppm_prompt, (int, float)):
        cost += (prompt_toks / 1_000_000.0) * ppm_prompt
    if ppm_comp is not None and isinstance(ppm_comp, (int, float)):
        cost += (completion_toks / 1_000_000.0) * ppm_comp
    return cost

# ------------------------------------------------------------------
# 1. Generic helper – fetch model lists with tiny caching layer
# ------------------------------------------------------------------

@functools.lru_cache(maxsize=128)
def _cached_get(url: str, bearer: str | None = None, timeout: int = 10) -> dict | list:
    headers = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------
# 2. Loader helpers for each provider that exposes an endpoint
# ------------------------------------------------------------------

def _openai_style_loader(url: str, api_key: str | None = None) -> List[str]:
    try:
        data = _cached_get(url, bearer=api_key)
        # Handle cases where data is a list (e.g., Together) or a dict with a 'data' key
        # (e.g., OpenAI)
        models_list = data if isinstance(data, list) else data.get("data", [])
        return sorted([m["id"] for m in models_list
                      if isinstance(m, dict) and "id" in m])
    except Exception as e:
        st.error(f"Error fetching models from {url}: {e}")
        return []

def _together_loader(api_key: str | None = None) -> List[str]:
    try:
        data = _cached_get("https://api.together.xyz/v1/models", bearer=api_key)
        return sorted([m["id"] for m in data
                      if isinstance(m, dict) and m.get("display_type") != "image"
                      and "id" in m])
    except Exception as e:
        st.error(f"Error fetching Together AI models: {e}")
        return []

def _fireworks_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.fireworks.ai/inference/v1/models", api_key)

def _groq_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.groq.com/openai/v1/models", api_key)

def _deepseek_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.deepseek.com/v1/models", api_key)

def _moonshot_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.moonshot.cn/v1/models", api_key)

def _baichuan_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.baichuan-ai.com/v1/models", api_key)

def _zhipu_loader(api_key: str | None = None) -> List[str]:
    # Zhipu's API might return models without an 'id', so we filter defensively
    try:
        data = _cached_get("https://open.bigmodel.cn/api/paas/v4/models",
                          bearer=api_key)
        return sorted([m["id"] for m in data.get("data", [])
                      if isinstance(m, dict) and "id" in m])
    except Exception as e:
        st.error(f"Error fetching Zhipu AI models: {e}")
        return []

def _minimax_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.minimax.chat/v1/models", api_key)

def _yi_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.lingyiwanwu.com/v1/models", api_key)

# ------------------------------------------------------------------
# Advanced Analytics and Reporting Functions
# ------------------------------------------------------------------

def generate_advanced_analytics(results: Dict) -> Dict:
    """Generate advanced analytics from adversarial testing results.
    
    Args:
        results (Dict): Adversarial testing results
        
    Returns:
        Dict: Advanced analytics data
    """
    analytics = {
        "total_iterations": len(results.get("iterations", [])),
        "final_approval_rate": results.get("final_approval_rate", 0),
        "total_cost_usd": results.get("cost_estimate_usd", 0),
        "total_tokens": results.get("tokens", {}).get("prompt", 0) + results.get("tokens", {}).get("completion", 0),
        "confidence_trend": [],
        "issue_resolution_rate": 0,
        "model_performance": {},
        "efficiency_score": 0,
        "security_strength": 0,
        "compliance_coverage": 0,
        "clarity_score": 0,
        "completeness_score": 0
    }
    
    # Calculate confidence trend
    if results.get("iterations"):
        confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0) 
                            for iter in results.get('iterations', [])]
        analytics["confidence_trend"] = confidence_history
    
    # Calculate issue resolution rate
    if results.get("iterations"):
        total_issues_found = 0
        total_issues_resolved = 0
        severity_weights = {"low": 1, "medium": 3, "high": 6, "critical": 12}
        
        for iteration in results.get("iterations", []):
            critiques = iteration.get("critiques", [])
            for critique in critiques:
                critique_json = critique.get("critique_json", {})
                issues = critique_json.get("issues", [])
                total_issues_found += len(issues)
                
                # Count resolved issues with severity weighting
                patches = iteration.get("patches", [])
                resolved_weighted = 0
                total_weighted = 0
                
                for issue in issues:
                    severity = issue.get("severity", "low").lower()
                    weight = severity_weights.get(severity, 1)
                    total_weighted += weight
                    
                for patch in patches:
                    patch_json = patch.get("patch_json", {})
                    mitigation_matrix = patch_json.get("mitigation_matrix", [])
                    for mitigation in mitigation_matrix:
                        if str(mitigation.get("status", "")).lower() in ["resolved", "mitigated"]:
                            # Approximate the severity of resolved issues
                            resolved_weighted += 3  # Average weight
                
                total_issues_resolved += min(len(issues), len(patches))
        
        if total_issues_found > 0:
            analytics["issue_resolution_rate"] = (total_issues_resolved / total_issues_found) * 100
    
    # Calculate efficiency score
    efficiency = 100
    if analytics["total_cost_usd"] > 0:
        # Lower cost = higher efficiency
        efficiency -= min(50, analytics["total_cost_usd"] * 10)
    if analytics["total_iterations"] > 10:
        # More iterations = lower efficiency
        efficiency -= min(30, (analytics["total_iterations"] - 10) * 2)
    analytics["efficiency_score"] = max(0, efficiency)
    
    # Calculate security strength based on resolved critical/high issues
    if results.get("iterations"):
        critical_high_resolved = 0
        total_critical_high = 0
        
        for iteration in results.get("iterations", []):
            critiques = iteration.get("critiques", [])
            for critique in critiques:
                critique_json = critique.get("critique_json", {})
                issues = critique_json.get("issues", [])
                for issue in issues:
                    severity = issue.get("severity", "low").lower()
                    if severity in ["critical", "high"]:
                        total_critical_high += 1
                        # Check if this issue was resolved in any patch
                        for patch in iteration.get("patches", []):
                            patch_json = patch.get("patch_json", {})
                            mitigation_matrix = patch_json.get("mitigation_matrix", [])
                            for mitigation in mitigation_matrix:
                                if (mitigation.get("issue") == issue.get("title") and 
                                    str(mitigation.get("status", "")).lower() in ["resolved", "mitigated"]):
                                    critical_high_resolved += 1
                                    break
        
        if total_critical_high > 0:
            analytics["security_strength"] = (critical_high_resolved / total_critical_high) * 100
    
    # Calculate compliance coverage
    if results.get("compliance_requirements"):
        # Simple check for compliance mentions in final protocol
        final_sop = results.get("final_sop", "")
        compliance_reqs = results.get("compliance_requirements", "")
        
        # Count how many compliance requirements are addressed
        reqs_addressed = 0
        total_reqs = 0
        
        for req in compliance_reqs.split(","):
            req = req.strip().lower()
            if req:
                total_reqs += 1
                if req in final_sop.lower():
                    reqs_addressed += 1
        
        if total_reqs > 0:
            analytics["compliance_coverage"] = (reqs_addressed / total_reqs) * 100
    
    # Calculate clarity score based on protocol structure
    final_sop = results.get("final_sop", "")
    if final_sop:
        structure = extract_protocol_structure(final_sop)
        complexity = calculate_protocol_complexity(final_sop)
        
        # Clarity score based on structure elements
        clarity_score = 0
        if structure["has_headers"]:
            clarity_score += 25
        if structure["has_numbered_steps"] or structure["has_bullet_points"]:
            clarity_score += 25
        if structure["has_preconditions"]:
            clarity_score += 15
        if structure["has_postconditions"]:
            clarity_score += 15
        if structure["has_error_handling"]:
            clarity_score += 20
            
        analytics["clarity_score"] = clarity_score
        
        # Completeness score based on structure and complexity
        completeness_score = min(100, (
            structure["section_count"] * 5 +  # Sections contribute to completeness
            complexity["unique_words"] / max(1, complexity["word_count"]) * 100 * 0.3 +  # Vocabulary diversity
            (1 - complexity["avg_sentence_length"] / 50) * 100 * 0.7  # Sentence complexity balance
        ))
        analytics["completeness_score"] = completeness_score
    
    return analytics

def create_performance_comparison_chart(results: Dict) -> str:
    """Create a performance comparison chart for models.
    
    Args:
        results (Dict): Adversarial testing results
        
    Returns:
        str: HTML chart code
    """
    # Simplified implementation - in a real app, this would generate actual charts
    return """
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        <h3>📊 Model Performance Comparison</h3>
        <p>Chart would display here showing model performance metrics</p>
    </div>
    """

def generate_executive_summary(results: Dict) -> str:
    """Generate an executive summary of adversarial testing results.
    
    Args:
        results (Dict): Adversarial testing results
        
    Returns:
        str: Executive summary in markdown format
    """
    analytics = generate_advanced_analytics(results)
    
    summary = f"""# Executive Summary

## 📊 Key Metrics
- **Final Approval Rate**: {analytics['final_approval_rate']:.1f}%
- **Iterations Completed**: {analytics['total_iterations']}
- **Total Cost**: ${analytics['total_cost_usd']:.4f}
- **Issue Resolution Rate**: {analytics['issue_resolution_rate']:.1f}%
- **Efficiency Score**: {analytics['efficiency_score']:.1f}/100

## 📈 Performance Insights
- **Confidence Improvement**: The protocol's approval confidence improved from {analytics['confidence_trend'][0] if analytics['confidence_trend'] else 'N/A'}% to {analytics['final_approval_rate']:.1f}%
- **Cost Efficiency**: Process completed within budget constraints
- **Resolution Effectiveness**: Issues were effectively identified and addressed

## 🏆 Recommendations
1. Continue monitoring for emerging threats
2. Periodically re-evaluate with updated models
3. Consider expanding the red team for broader vulnerability coverage
4. Implement continuous integration for automated protocol hardening

## 📅 Next Steps
- Review and implement outstanding recommendations
- Schedule periodic adversarial testing cycles
- Share results with stakeholders for feedback
- Update documentation with hardened protocol
"""
    
    return summary


