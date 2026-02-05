#!/usr/bin/env python3
"""
Centralized Messaging System for Knowledge Extraction Agent
Provides clean, visually appealing console output with progress indicators.
"""

import sys
import time
from typing import Optional, Dict, Any
from datetime import datetime
import logging

class ConsoleMessenger:
    """Centralized console messaging with visual appeal and progress tracking"""
    
    # ANSI color codes for terminal output
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bg_blue': '\033[44m',
        'bg_green': '\033[42m',
        'bg_yellow': '\033[43m',
        'bg_red': '\033[41m'
    }
    
    # Unicode symbols for visual appeal
    SYMBOLS = {
        'success': '[OK]',
        'error': '[FAIL]',
        'warning': '[WARN]',
        'info': 'ℹ️',
        'progress': '🔄',
        'document': '📄',
        'extraction': '🔍',
        'processing': '⚙️',
        'complete': '🎯',
        'arrow': '➤',
        'bullet': '*',
        'check': '[OK]',
        'cross': '[FAIL]',
        'spinner': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.current_operation = None
        self.start_time = None
        self.spinner_index = 0
        
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if terminal supports it"""
        if not sys.stdout.isatty():
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _format_time(self, seconds: float) -> str:
        """Format elapsed time in a readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"
    
    def _print_header(self, title: str, color: str = 'blue'):
        """Print a formatted header"""
        width = 60
        print()
        print(self._colorize('═' * width, color))
        print(self._colorize(f" {title} ".center(width), color))
        print(self._colorize('═' * width, color))
        print()
    
    def _print_section(self, title: str, color: str = 'cyan'):
        """Print a formatted section header"""
        print()
        print(self._colorize(f"{self.SYMBOLS['arrow']} {title}", color))
        print(self._colorize('─' * (len(title) + 4), 'dim'))
    
    def start_operation(self, operation: str, details: Optional[str] = None):
        """Start tracking a new operation"""
        self.current_operation = operation
        self.start_time = time.time()
        
        if details:
            print(f"{self.SYMBOLS['progress']} {self._colorize(operation, 'blue')}: {details}")
        else:
            print(f"{self.SYMBOLS['progress']} {self._colorize(operation, 'blue')}")
    
    def update_operation(self, details: str):
        """Update current operation with new details"""
        if self.current_operation:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"   {self.SYMBOLS['bullet']} {details} {self._colorize(f'({self._format_time(elapsed)})', 'dim')}")
    
    def complete_operation(self, success: bool = True, details: Optional[str] = None):
        """Complete current operation"""
        if not self.current_operation:
            return
            
        elapsed = time.time() - self.start_time if self.start_time else 0
        symbol = self.SYMBOLS['success'] if success else self.SYMBOLS['error']
        color = 'green' if success else 'red'
        
        if details:
            print(f"{symbol} {self._colorize(self.current_operation, color)}: {details} {self._colorize(f'({self._format_time(elapsed)})', 'dim')}")
        else:
            print(f"{symbol} {self._colorize(self.current_operation, color)} {self._colorize(f'({self._format_time(elapsed)})', 'dim')}")
        
        self.current_operation = None
        self.start_time = None
    
    def info(self, message: str, details: Optional[str] = None):
        """Print informational message"""
        if details:
            print(f"{self.SYMBOLS['info']} {message}: {details}")
        else:
            print(f"{self.SYMBOLS['info']} {message}")
    
    def success(self, message: str, details: Optional[str] = None):
        """Print success message"""
        if details:
            print(f"{self.SYMBOLS['success']} {self._colorize(message, 'green')}: {details}")
        else:
            print(f"{self.SYMBOLS['success']} {self._colorize(message, 'green')}")
    
    def warning(self, message: str, details: Optional[str] = None):
        """Print warning message"""
        if details:
            print(f"{self.SYMBOLS['warning']} {self._colorize(message, 'yellow')}: {details}")
        else:
            print(f"{self.SYMBOLS['warning']} {self._colorize(message, 'yellow')}")
    
    def error(self, message: str, details: Optional[str] = None):
        """Print error message"""
        if details:
            print(f"{self.SYMBOLS['error']} {self._colorize(message, 'red')}: {details}")
        else:
            print(f"{self.SYMBOLS['error']} {self._colorize(message, 'red')}")
    
    def document_processing(self, filename: str, method: str, total: int, current: int):
        """Print document processing status"""
        progress = f"({current}/{total})"
        print(f"{self.SYMBOLS['document']} {self._colorize('Processing', 'blue')} {filename} {self._colorize(f'[{method}]', 'cyan')} {self._colorize(progress, 'dim')}")
    
    def extraction_progress(self, stage: str, documents_count: int, results_count: int):
        """Print extraction progress"""
        print(f"{self.SYMBOLS['extraction']} {self._colorize(stage, 'blue')}: {documents_count} docs -> {results_count} results")
    
    def parsing_method(self, method: str, filename: str):
        """Print parsing method being used"""
        method_display = "Docling Parsing" if method == "docling" else "Fast Parsing"
        print(f"{self.SYMBOLS['processing']} {self._colorize(method_display, 'cyan')}: {filename}")
    
    def model_generation(self, task_name: str, fields_count: int):
        """Print model generation status"""
        print(f"{self.SYMBOLS['processing']} {self._colorize('Generating Models', 'blue')}: {task_name} ({fields_count} fields)")
    
    def results_summary(self, results: Dict[str, Any], extraction_type: str):
        """Print extraction results summary"""
        self._print_section("Extraction Results Summary", 'green')
        
        if extraction_type == 'hierarchical':
            # Case 2 results
            consolidated_count = len(results.get('extraction_results', []))
            stage_count = len(results.get('stage_results', {}))
            docs_processed = results.get('processing_metadata', {}).get('total_documents', 0)
            
            print(f"   {self.SYMBOLS['document']} Documents Processed: {self._colorize(str(docs_processed), 'green')}")
            print(f"   {self.SYMBOLS['extraction']} Extraction Stages: {self._colorize(str(stage_count), 'green')}")
            print(f"   {self.SYMBOLS['complete']} Consolidated Results: {self._colorize(str(consolidated_count), 'green')}")
            
            # Show stage breakdown
            if results.get('stage_results'):
                print(f"   {self.SYMBOLS['bullet']} Stage Breakdown:")
                for stage_name, stage_data in results['stage_results'].items():
                    print(f"      {self.SYMBOLS['check']} {stage_name}: {len(stage_data)} records")
        else:
            # Case 1 or single-type results
            results_count = len(results) if isinstance(results, list) else 0
            print(f"   {self.SYMBOLS['complete']} Extracted Records: {self._colorize(str(results_count), 'green')}")
    
    def debug(self, message: str, details: Optional[str] = None):
        """Print debug message (only if verbose mode is enabled)"""
        if self.verbose:
            if details:
                print(f"{self._colorize('DEBUG', 'dim')}: {message}: {details}")
            else:
                print(f"{self._colorize('DEBUG', 'dim')}: {message}")
    
    def spinner_update(self):
        """Update spinner animation"""
        if not sys.stdout.isatty():
            return
            
        self.spinner_index = (self.spinner_index + 1) % len(self.SYMBOLS['spinner'])
        spinner = self.SYMBOLS['spinner'][self.spinner_index]
        print(f"\r{spinner} Processing...", end='', flush=True)
    
    def clear_line(self):
        """Clear current line"""
        if sys.stdout.isatty():
            print('\r' + ' ' * 80 + '\r', end='', flush=True)

# Global messenger instance
messenger = ConsoleMessenger(verbose=False)

def set_verbose_mode(verbose: bool):
    """Enable or disable verbose debug output"""
    global messenger
    messenger.verbose = verbose

def get_messenger() -> ConsoleMessenger:
    """Get the global messenger instance"""
    return messenger
