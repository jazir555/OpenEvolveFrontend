#!/usr/bin/env python3
"""
Scan Python files for import errors - Batch 10
"""

import ast
import json
import py_compile
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import os


def extract_imports_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Extract all imports from a Python file using AST."""
    imports = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'asname': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'asname': alias.asname,
                        'level': level,
                        'line': node.lineno
                    })
    except Exception as e:
        pass
    
    return imports


def check_syntax_with_ast(filepath: str) -> Optional[Dict[str, Any]]:
    """Check for syntax errors using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        ast.parse(content)
        return None
    except SyntaxError as e:
        return {
            'error_type': 'syntax_error',
            'line_number': e.lineno if e.lineno else 0,
            'message': str(e),
            'suggested_fix': f"Check syntax around line {e.lineno}: {e.text}" if e.text else "Fix syntax error"
        }
    except Exception as e:
        return {
            'error_type': 'other',
            'line_number': 0,
            'message': f"Failed to parse: {str(e)}",
            'suggested_fix': "Check file encoding and content"
        }


def check_compile_with_py_compile(filepath: str) -> Optional[Dict[str, Any]]:
    """Check for compilation errors using py_compile."""
    try:
        py_compile.compile(filepath, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return {
            'error_type': 'syntax_error',
            'line_number': getattr(e, 'lineno', 0) or 0,
            'message': str(e),
            'suggested_fix': "Fix compilation error"
        }


def check_relative_imports(filepath: str, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check if relative imports can resolve."""
    errors = []
    file_path = Path(filepath)
    file_dir = file_path.parent
    
    for imp in imports:
        if imp['type'] == 'from_import' and imp['level'] > 0:
            # This is a relative import
            level = imp['level']
            module = imp['module']
            
            # Calculate target directory
            target_dir = file_dir
            for _ in range(level - 1):
                target_dir = target_dir.parent
            
            # Check if the module exists
            if module:
                module_parts = module.split('.')
                current_path = target_dir
                
                for part in module_parts:
                    current_path = current_path / part
                
                # Check as package
                pkg_path = current_path / '__init__.py'
                mod_path = current_path.with_suffix('.py')
                
                if not pkg_path.exists() and not mod_path.exists():
                    errors.append({
                        'error_type': 'import_error',
                        'line_number': imp['line'],
                        'message': f"Relative import cannot resolve: {'.' * level}{module}",
                        'suggested_fix': f"Ensure module '{module}' exists relative to file"
                    })
    
    return errors


def check_common_import_typos(imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check for common typos in module names."""
    errors = []
    
    # Common typos mapping
    common_modules = {
        'numpy': ['nmpy', 'numy', 'nump', 'numpy'],
        'pandas': ['pands', 'panda', 'pandass', 'pdas'],
        'matplotlib': ['matplot', 'matplotlib', 'matplotib', 'matplotlb'],
        'sklearn': ['sklean', 'sklern', 'sklarn', 'skleran'],
        'tensorflow': ['tensorflw', 'tensorflwo', 'tensrflow', 'tensor_flow'],
        'torch': ['toch', 'troch', 'torh', 'pytorch'],
        'requests': ['request', 'reqests', 'reqeusts', 'requets'],
        'flask': ['flsk', 'flas', 'flasck'],
        'django': ['djnago', 'djang', 'djngo', 'djano'],
        'fastapi': ['fastpi', 'fast_app', 'fast_api', 'fastappi'],
        'pydantic': ['pydantic', 'pydatic', 'pydanitc', 'pydntic'],
        'sqlalchemy': ['sqlachemy', 'sqlalchmy', 'sqlalcemy', 'sqlalchmey'],
        'asyncio': ['asynco', 'asynio', 'async_io', 'asynci'],
        'datetime': ['datetme', 'datetim', 'date_time', 'datettime'],
        'json': ['jso', 'josn', 'jsson'],
        'os': ['oss', 'so'],
        'sys': ['syss', 'system', 'ssy'],
        'typing': ['typin', 'tying', 'typng', 'typeing'],
        'pathlib': ['pathlb', 'pathli', 'path_lib', 'pthlib'],
        'collections': ['collectins', 'collectons', 'collection', 'collectios'],
        'itertools': ['iterools', 'itertols', 'itertool', 'iter_tools'],
        'functools': ['functols', 'functool', 'functuils', 'func_tools'],
        'inspect': ['inspec', 'inpect', 'inspeect', 'inspct'],
        'hashlib': ['hashlb', 'hashli', 'hash_lib', 'hshlib'],
        're': ['ree', 'regex', 'regx'],
        'math': ['maths', 'mth', 'mat'],
        'random': ['randm', 'radom', 'randon', 'ramdom'],
        'uuid': ['uiid', 'uud', 'uu_id'],
        'base64': ['base_64', 'bse64', 'bas64', 'base4'],
        'copy': ['cpy', 'coy', 'cop'],
        'pickle': ['picle', 'pickl', 'pikle', 'picklee'],
        'csv': ['cs', 'cvs', 'ccsv'],
        'yaml': ['yml', 'yam', 'yamm', 'yamll'],
        'toml': ['tom', 'tomll', 'toml'],
        'zipfile': ['zipile', 'zip_file', 'zipflie', 'zippfile'],
        'tarfile': ['tarile', 'tar_file', 'tarflie', 'tarrfile'],
        'gzip': ['gip', 'gizp', 'gzipp', 'ggzip'],
        'shutil': ['shutil', 'sutil', 'shutil', 'shutil'],
        'subprocess': ['subproces', 'subprocess', 'subprocss', 'sub_proces'],
        'tempfile': ['tempile', 'temp_file', 'tempflie', 'tempfilee'],
        'io': ['iio', 'inout', 'input_output'],
        'warnings': ['warning', 'warnigs', 'warnngs', 'warings'],
        'logging': ['loging', 'loggin', 'loggng', 'loging'],
        'traceback': ['tracebak', 'tracebck', 'trace_back', 'tracebakk'],
        'unittest': ['unitest', 'unittst', 'unit_test', 'unitttest'],
        'pytest': ['pytst', 'pytes', 'py_test', 'pyttest'],
        'mock': ['mok', 'moch', 'mokk', 'mokc'],
        'argparse': ['argpar', 'arg_pars', 'argparsee', 'argparse'],
        'configparser': ['configpar', 'config_pars', 'configparse', 'configparsr'],
        'socket': ['sockt', 'soket', 'socet', 'sockket'],
        'urllib': ['urlib', 'urllibb', 'url_lib', 'urllb'],
        'http': ['htt', 'htp', 'httpp', 'htttp'],
        'ftplib': ['ftlib', 'ftplb', 'ftp_lib', 'ftplibb'],
        'smtplib': ['smtlib', 'smtplb', 'smtp_lib', 'smtplibb'],
        'email': ['emal', 'emial', 'emaill', 'e_mail'],
        'html': ['htm', 'htl', 'htmll', 'hml'],
        'xml': ['xm', 'xmll', 'xnl', 'xxml'],
        'json': ['jso', 'josn', 'jsson', 'jsonn'],
        'csv': ['cs', 'cvs', 'ccsv', 'csvv'],
        'sqlite3': ['sqlite', 'sqlit', 'sqlte3', 'sqllite3'],
        'threading': ['threadng', 'threadin', 'threding', 'threadig'],
        'multiprocessing': ['multprocesing', 'multiprocesing', 'multi_processing', 'multiprocessng'],
        'concurrent': ['concur', 'concurr', 'conncurrent', 'concurernt'],
        'queue': ['que', 'queu', 'quee', 'quue'],
        'weakref': ['weakf', 'weak_ef', 'weakre', 'weakref'],
        'gc': ['g', 'gcc', 'gcollect', 'garbage'],
        'atexit': ['ateit', 'at_exit', 'atext', 'ateixt'],
        'signal': ['signl', 'signa', 'signaal', 'signall'],
        'contextlib': ['contextlb', 'context_lib', 'contextli', 'contextlibb'],
        'functools': ['functols', 'functool', 'functuils', 'func_tools'],
        'operator': ['operatr', 'operat', 'opertor', 'operatorr'],
        'enum': ['enm', 'enumn', 'enumm', 'ennum'],
        'dataclasses': ['dataclas', 'data_classes', 'dataclass', 'dataclassess'],
        'typing': ['typin', 'tying', 'typng', 'typeing'],
        'collections': ['collectins', 'collectons', 'collection', 'collectios'],
        'abc': ['acb', 'aabc', 'abcc', 'abstract'],
        'numbers': ['numbes', 'number', 'numbrs', 'nnumbers'],
        'decimal': ['decial', 'deciml', 'decima', 'decimall'],
        'fractions': ['fractons', 'fractios', 'fraction', 'fractio'],
        'statistics': ['statstics', 'statistic', 'statstics', 'statisics'],
        'itertools': ['iterools', 'itertols', 'itertool', 'iter_tools'],
        'bisect': ['bisct', 'bisec', 'bisett', 'bisecct'],
        'heapq': ['heap', 'heap_que', 'heappq', 'heapqq'],
        'copy': ['cpy', 'coy', 'cop', 'copyy'],
        'pprint': ['pprintt', 'ppint', 'pprnt', 'pp_print'],
        'reprlib': ['reprlb', 'repr_lib', 'reprli', 'reprlibb'],
        'string': ['strng', 'sting', 'srting', 'stringg'],
        're': ['ree', 'regex', 'regx', 're'],
        'difflib': ['difflb', 'diff_lib', 'diffl', 'difflibb'],
        'textwrap': ['textwap', 'text_wrap', 'texwrap', 'textwrapp'],
        'unicodedata': ['unicodedat', 'unicode_data', 'unicodedta', 'unicodedataa'],
        'stringprep': ['stringrep', 'string_prep', 'stringprp', 'stringprepp'],
        'readline': ['readlin', 'read_line', 'readine', 'readlinee'],
        'rlcompleter': ['rlcomplet', 'rl_completer', 'rlcompletr', 'rlcompleterr'],
    }
    
    # Build reverse lookup for typos
    typo_to_correct = {}
    for correct, typos in common_modules.items():
        for typo in typos:
            if typo != correct:
                typo_to_correct[typo] = correct
    
    for imp in imports:
        module_name = imp.get('module', '') or imp.get('name', '')
        if module_name in typo_to_correct:
            errors.append({
                'error_type': 'import_error',
                'line_number': imp['line'],
                'message': f"Possible typo in import: '{module_name}' should be '{typo_to_correct[module_name]}'",
                'suggested_fix': f"Change '{module_name}' to '{typo_to_correct[module_name]}'"
            })
    
    return errors


def scan_file(filepath: str) -> List[Dict[str, Any]]:
    """Scan a single file for import errors."""
    errors = []
    
    # Check if file exists
    if not os.path.exists(filepath):
        return [{
            'file': filepath,
            'error_type': 'other',
            'line_number': 0,
            'message': 'File does not exist',
            'suggested_fix': 'Remove from batch or locate file'
        }]
    
    # Check syntax with AST
    syntax_error = check_syntax_with_ast(filepath)
    if syntax_error:
        errors.append({
            'file': filepath,
            **syntax_error
        })
        return errors  # Don't continue if syntax error
    
    # Check compilation
    compile_error = check_compile_with_py_compile(filepath)
    if compile_error:
        errors.append({
            'file': filepath,
            **compile_error
        })
    
    # Extract imports
    imports = extract_imports_from_file(filepath)
    
    # Check relative imports
    relative_errors = check_relative_imports(filepath, imports)
    for err in relative_errors:
        errors.append({
            'file': filepath,
            **err
        })
    
    # Check for typos
    typo_errors = check_common_import_typos(imports)
    for err in typo_errors:
        errors.append({
            'file': filepath,
            **err
        })
    
    return errors


def main():
    """Main function to scan batch 10."""
    batch_file = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend\batch_10.txt'
    output_file = r'c:\Users\mmeadow\Documents\OpenEvolve\Frontend\import_errors_batch_10.json'
    
    # Read file list
    with open(batch_file, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"Scanning {len(files)} files...")
    
    all_errors = []
    processed = 0
    
    for filepath in files:
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(files)} files...")
        
        try:
            file_errors = scan_file(filepath)
            all_errors.extend(file_errors)
        except Exception as e:
            all_errors.append({
                'file': filepath,
                'error_type': 'other',
                'line_number': 0,
                'message': f"Scanner error: {str(e)}",
                'suggested_fix': "Manual inspection required"
            })
    
    # Create report
    report = {
        'total_files': len(files),
        'errors_found': len(all_errors),
        'errors': all_errors
    }
    
    # Write JSON report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nScan complete!")
    print(f"Total files scanned: {len(files)}")
    print(f"Errors found: {len(all_errors)}")
    print(f"Report saved to: {output_file}")


if __name__ == '__main__':
    main()
