#!/usr/bin/env python3
"""
BubbleLab Gap Analysis Script
Analyzes all bubbles to identify implementation gaps
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

class BubbleAnalyzer:
    def __init__(self, bubbles_dir: str):
        self.bubbles_dir = Path(bubbles_dir)
        self.results = {
            'service_bubbles': [],
            'tool_bubbles': [],
            'workflow_bubbles': []
        }
        self.stats = {
            'total_bubbles': 0,
            'total_methods': 0,
            'implemented_methods': 0,
            'placeholder_methods': 0,
            'missing_implementations': 0
        }
        self.gaps = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }

    def is_main_bubble_file(self, file_path: Path) -> bool:
        """Check if this is a main bubble implementation file"""
        name = file_path.name
        # Exclude test files, schema files, index files, and utility files
        if any(x in name for x in ['test', 'spec', 'schema', 'index', 'utils', 'template', 'flow']):
            return False
        # Exclude subdirectories that are not main bubbles
        if 'apify/actors' in str(file_path):
            return False
        if 'google-sheets/' in str(file_path) and name != 'google-sheets.ts':
            return False
        if 'notion/' in str(file_path) and name != 'notion.ts':
            return False
        return True

    def extract_bubble_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract bubble information from a file"""
        content = file_path.read_text(encoding='utf-8', errors='ignore')

        # Extract bubble name/type
        bubble_type = None
        if 'service-bubble' in str(file_path):
            bubble_type = 'service'
        elif 'tool-bubble' in str(file_path):
            bubble_type = 'tool'
        elif 'workflow-bubble' in str(file_path):
            bubble_type = 'workflow'

        # Extract bubble name from file
        bubble_name = file_path.stem.replace('-bubble', '').replace('.workflow', '')

        # Extract methods and their implementations
        methods = self.extract_methods(content)

        # Check for specific issues
        issues = self.analyze_issues(content, bubble_name)

        return {
            'name': bubble_name,
            'type': bubble_type,
            'file_path': str(file_path),
            'methods': methods,
            'issues': issues,
            'line_count': len(content.split('\n'))
        }

    def extract_methods(self, content: str) -> List[Dict[str, Any]]:
        """Extract all methods from the content"""
        methods = []
        # Match async/non-async function definitions
        pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)|(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*=>'

        matches = re.finditer(pattern, content)
        for match in matches:
            method_name = match.group(1) or match.group(2)
            if method_name:
                # Find the implementation
                method_start = match.start()
                implementation = self.get_implementation(content, method_start)

                is_placeholder = self.is_placeholder(implementation)
                is_empty = self.is_empty_implementation(implementation)

                methods.append({
                    'name': method_name,
                    'implementation': implementation[:200],  # First 200 chars
                    'is_placeholder': is_placeholder,
                    'is_empty': is_empty,
                    'has_error_handling': 'try' in implementation or 'catch' in implementation,
                    'has_timeout': 'timeout' in implementation.lower(),
                    'has_retry': 'retry' in implementation.lower(),
                    'has_validation': 'validate' in implementation.lower() or 'check' in implementation.lower()
                })

        return methods

    def get_implementation(self, content: str, start: int) -> str:
        """Extract the implementation of a method"""
        # Find the opening brace
        brace_start = content.find('{', start)
        if brace_start == -1:
            return ''

        # Find matching closing brace
        depth = 1
        i = brace_start + 1
        while i < len(content) and depth > 0:
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
            i += 1

        return content[brace_start:i]

    def is_placeholder(self, implementation: str) -> bool:
        """Check if implementation is a placeholder"""
        placeholder_indicators = [
            'TODO',
            'FIXME',
            'NotImplemented',
            'throw new Error',
            'placeholder',
            '// implement',
            '/* implement'
        ]
        impl_lower = implementation.lower()
        return any(indicator.lower() in impl_lower for indicator in placeholder_indicators)

    def is_empty_implementation(self, implementation: str) -> bool:
        """Check if implementation is effectively empty"""
        # Remove whitespace and common empty patterns
        cleaned = re.sub(r'\s+', '', implementation)
        empty_patterns = ['{}', '{return;}', '{returnnull;}', '{returnundefined;}']
        return cleaned in empty_patterns or len(cleaned) < 10

    def analyze_issues(self, content: str, bubble_name: str) -> List[Dict[str, Any]]:
        """Analyze specific issues in the bubble"""
        issues = []

        # Check for API calls without timeout
        api_calls = re.findall(r'(fetch|axios|request)\(', content)
        if api_calls and 'timeout' not in content.lower():
            issues.append({
                'severity': 'high',
                'type': 'missing_timeout',
                'message': f'API calls without timeout handling'
            })

        # Check for missing error handling
        has_async = 'async' in content
        has_try_catch = 'try' in content and 'catch' in content
        if has_async and not has_try_catch:
            issues.append({
                'severity': 'medium',
                'type': 'missing_error_handling',
                'message': f'Async functions without try-catch blocks'
            })

        # Check for hardcoded credentials
        if re.search(r'(api_key|password|secret)\s*=\s*["\']', content):
            issues.append({
                'severity': 'critical',
                'type': 'hardcoded_credentials',
                'message': f'Potentially hardcoded credentials detected'
            })

        # Check for missing input validation
        has_params = 'params' in content or 'input' in content or 'args' in content
        has_validation = 'validate' in content.lower() or 'zod' in content.lower() or 'joi' in content.lower()
        if has_params and not has_validation:
            issues.append({
                'severity': 'high',
                'type': 'missing_validation',
                'message': f'Missing input validation'
            })

        # Check for TODO comments
        todos = re.findall(r'TODO|FIXME|XXX', content)
        if todos:
            issues.append({
                'severity': 'medium',
                'type': 'pending_work',
                'message': f'{len(todos)} TODO/FIXME comments found'
            })

        # Check for placeholder returns
        if 'TODO' in content or 'NotImplemented' in content:
            issues.append({
                'severity': 'critical',
                'type': 'placeholder_return',
                'message': f'Contains placeholder implementations'
            })

        return issues

    def analyze(self):
        """Analyze all bubbles"""
        # Analyze service bubbles
        service_dir = self.bubbles_dir / 'service-bubble'
        if service_dir.exists():
            for file_path in service_dir.rglob('*.ts'):
                if self.is_main_bubble_file(file_path):
                    bubble_info = self.extract_bubble_info(file_path)
                    self.results['service_bubbles'].append(bubble_info)

        # Analyze tool bubbles
        tool_dir = self.bubbles_dir / 'tool-bubble'
        if tool_dir.exists():
            for file_path in tool_dir.rglob('*.ts'):
                if self.is_main_bubble_file(file_path):
                    bubble_info = self.extract_bubble_info(file_path)
                    self.results['tool_bubbles'].append(bubble_info)

        # Analyze workflow bubbles
        workflow_dir = self.bubbles_dir / 'workflow-bubble'
        if workflow_dir.exists():
            for file_path in workflow_dir.rglob('*.ts'):
                if self.is_main_bubble_file(file_path):
                    bubble_info = self.extract_bubble_info(file_path)
                    self.results['workflow_bubbles'].append(bubble_info)

        # Calculate statistics
        self.calculate_stats()
        self.categorize_gaps()

    def calculate_stats(self):
        """Calculate overall statistics"""
        all_bubbles = (self.results['service_bubbles'] +
                      self.results['tool_bubbles'] +
                      self.results['workflow_bubbles'])

        self.stats['total_bubbles'] = len(all_bubbles)

        for bubble in all_bubbles:
            for method in bubble['methods']:
                self.stats['total_methods'] += 1
                if not method['is_placeholder'] and not method['is_empty']:
                    self.stats['implemented_methods'] += 1
                else:
                    self.stats['placeholder_methods'] += 1

        self.stats['missing_implementations'] = self.stats['placeholder_methods']

    def categorize_gaps(self):
        """Categorize gaps by severity"""
        all_bubbles = (self.results['service_bubbles'] +
                      self.results['tool_bubbles'] +
                      self.results['workflow_bubbles'])

        for bubble in all_bubbles:
            for issue in bubble['issues']:
                gap = {
                    'bubble': bubble['name'],
                    'type': bubble['type'],
                    'issue_type': issue['type'],
                    'message': issue['message'],
                    'file_path': bubble['file_path']
                }
                self.gaps[issue['severity']].append(gap)

            # Check for method-level gaps
            for method in bubble['methods']:
                if method['is_placeholder']:
                    self.gaps['critical'].append({
                        'bubble': bubble['name'],
                        'type': bubble['type'],
                        'issue_type': 'placeholder_method',
                        'message': f'Method "{method["name"]}" is a placeholder',
                        'file_path': bubble['file_path']
                    })

                if method['is_empty']:
                    self.gaps['high'].append({
                        'bubble': bubble['name'],
                        'type': bubble['type'],
                        'issue_type': 'empty_method',
                        'message': f'Method "{method["name"]}" is empty',
                        'file_path': bubble['file_path']
                    })

                if not method['has_error_handling']:
                    self.gaps['medium'].append({
                        'bubble': bubble['name'],
                        'type': bubble['type'],
                        'issue_type': 'missing_error_handling',
                        'message': f'Method "{method["name"]}" missing error handling',
                        'file_path': bubble['file_path']
                    })

                if not method['has_timeout']:
                    self.gaps['low'].append({
                        'bubble': bubble['name'],
                        'type': bubble['type'],
                        'issue_type': 'missing_timeout',
                        'message': f'Method "{method["name"]}" missing timeout',
                        'file_path': bubble['file_path']
                    })

    def generate_report(self) -> str:
        """Generate a comprehensive gap analysis report"""
        report = []
        report.append("# BubbleLab Gap Analysis Report")
        report.append("")
        report.append(f"**Generated:** {self.get_timestamp()}")
        report.append("")

        # Summary Statistics
        report.append("## 1. Summary Statistics")
        report.append("")
        report.append(f"- **Total Bubbles:** {self.stats['total_bubbles']}")
        report.append(f"- **Service Bubbles:** {len(self.results['service_bubbles'])}")
        report.append(f"- **Tool Bubbles:** {len(self.results['tool_bubbles'])}")
        report.append(f"- **Workflow Bubbles:** {len(self.results['workflow_bubbles'])}")
        report.append("")
        report.append(f"- **Total Methods:** {self.stats['total_methods']}")
        report.append(f"- **Implemented Methods:** {self.stats['implemented_methods']}")
        report.append(f"- **Placeholder Methods:** {self.stats['placeholder_methods']}")
        report.append("")

        if self.stats['total_methods'] > 0:
            completion_rate = (self.stats['implemented_methods'] / self.stats['total_methods']) * 100
            report.append(f"- **Completion Rate:** {completion_rate:.1f}%")
        report.append("")

        # Critical Gaps
        report.append("## 2. Critical Gaps (Must Fix)")
        report.append("")
        if self.gaps['critical']:
            for gap in self.gaps['critical']:
                report.append(f"### {gap['bubble']} ({gap['type']})")
                report.append(f"- **Issue:** {gap['message']}")
                report.append(f"- **Type:** {gap['issue_type']}")
                report.append(f"- **File:** {gap['file_path']}")
                report.append("")
        else:
            report.append("No critical gaps found!")
            report.append("")

        # High Priority Gaps
        report.append("## 3. High Priority Gaps")
        report.append("")
        if self.gaps['high']:
            for gap in self.gaps['high'][:20]:  # Limit to first 20
                report.append(f"### {gap['bubble']} ({gap['type']})")
                report.append(f"- **Issue:** {gap['message']}")
                report.append(f"- **Type:** {gap['issue_type']}")
                report.append(f"- **File:** {gap['file_path']}")
                report.append("")

            if len(self.gaps['high']) > 20:
                report.append(f"... and {len(self.gaps['high']) - 20} more high priority gaps")
                report.append("")
        else:
            report.append("No high priority gaps found!")
            report.append("")

        # Medium Priority Gaps
        report.append("## 4. Medium Priority Gaps")
        report.append("")
        if self.gaps['medium']:
            report.append(f"**Total Medium Priority Issues:** {len(self.gaps['medium'])}")
            report.append("")
            # Group by type
            grouped = defaultdict(list)
            for gap in self.gaps['medium']:
                grouped[gap['issue_type']].append(gap)

            for issue_type, gaps in grouped.items():
                report.append(f"### {issue_type.replace('_', ' ').title()}: {len(gaps)} occurrences")
                report.append("")
        else:
            report.append("No medium priority gaps found!")
            report.append("")

        # Low Priority Gaps
        report.append("## 5. Low Priority Gaps")
        report.append("")
        report.append(f"**Total Low Priority Issues:** {len(self.gaps['low'])}")
        report.append("")

        # Implementation Roadmap
        report.append("## 6. Implementation Roadmap")
        report.append("")

        # Phase 1: Critical
        report.append("### Phase 1: Critical Fixes (Week 1)")
        report.append("**Priority:** P0 - Blocking production deployment")
        report.append("")
        critical_by_type = defaultdict(list)
        for gap in self.gaps['critical']:
            critical_by_type[gap['issue_type']].append(gap)

        for issue_type, gaps in critical_by_type.items():
            report.append(f"- **{issue_type}:** {len(gaps)} items")
            report.append(f"  - Estimated effort: {len(gaps) * 2} hours")
            report.append(f"  - Actions: Fix all placeholder implementations")
        report.append("")

        # Phase 2: High Priority
        report.append("### Phase 2: High Priority Fixes (Week 2)")
        report.append("**Priority:** P1 - Important for reliability")
        report.append("")
        high_by_type = defaultdict(list)
        for gap in self.gaps['high']:
            high_by_type[gap['issue_type']].append(gap)

        for issue_type, gaps in high_by_type.items():
            report.append(f"- **{issue_type}:** {len(gaps)} items")
            report.append(f"  - Estimated effort: {len(gaps)} hours")
            report.append(f"  - Actions: Implement empty methods, add timeouts")
        report.append("")

        # Phase 3: Medium Priority
        report.append("### Phase 3: Medium Priority Fixes (Week 3)")
        report.append("**Priority:** P2 - Important for production readiness")
        report.append("")
        medium_by_type = defaultdict(list)
        for gap in self.gaps['medium']:
            medium_by_type[gap['issue_type']].append(gap)

        for issue_type, gaps in medium_by_type.items():
            report.append(f"- **{issue_type}:** {len(gaps)} items")
            report.append(f"  - Estimated effort: {len(gaps) * 0.5} hours")
            report.append(f"  - Actions: Add error handling and validation")
        report.append("")

        # Detailed Bubble Analysis
        report.append("## 7. Detailed Bubble Analysis")
        report.append("")

        # Service Bubbles
        report.append("### 7.1 Service Bubbles")
        report.append("")
        for bubble in sorted(self.results['service_bubbles'], key=lambda x: x['name']):
            placeholder_count = sum(1 for m in bubble['methods'] if m['is_placeholder'])
            empty_count = sum(1 for m in bubble['methods'] if m['is_empty'])
            issue_count = len(bubble['issues'])

            status = "✅ Complete" if placeholder_count == 0 and empty_count == 0 else "⚠️ Incomplete"

            report.append(f"#### {bubble['name']} {status}")
            report.append(f"- **Methods:** {len(bubble['methods'])}")
            report.append(f"- **Placeholders:** {placeholder_count}")
            report.append(f"- **Empty Methods:** {empty_count}")
            report.append(f"- **Issues:** {issue_count}")
            if bubble['issues']:
                for issue in bubble['issues'][:3]:
                    report.append(f"  - {issue['severity'].upper()}: {issue['message']}")
            report.append("")

        # Tool Bubbles
        report.append("### 7.2 Tool Bubbles")
        report.append("")
        for bubble in sorted(self.results['tool_bubbles'], key=lambda x: x['name']):
            placeholder_count = sum(1 for m in bubble['methods'] if m['is_placeholder'])
            empty_count = sum(1 for m in bubble['methods'] if m['is_empty'])
            issue_count = len(bubble['issues'])

            status = "✅ Complete" if placeholder_count == 0 and empty_count == 0 else "⚠️ Incomplete"

            report.append(f"#### {bubble['name']} {status}")
            report.append(f"- **Methods:** {len(bubble['methods'])}")
            report.append(f"- **Placeholders:** {placeholder_count}")
            report.append(f"- **Empty Methods:** {empty_count}")
            report.append(f"- **Issues:** {issue_count}")
            if bubble['issues']:
                for issue in bubble['issues'][:3]:
                    report.append(f"  - {issue['severity'].upper()}: {issue['message']}")
            report.append("")

        # Workflow Bubbles
        report.append("### 7.3 Workflow Bubbles")
        report.append("")
        for bubble in sorted(self.results['workflow_bubbles'], key=lambda x: x['name']):
            placeholder_count = sum(1 for m in bubble['methods'] if m['is_placeholder'])
            empty_count = sum(1 for m in bubble['methods'] if m['is_empty'])
            issue_count = len(bubble['issues'])

            status = "✅ Complete" if placeholder_count == 0 and empty_count == 0 else "⚠️ Incomplete"

            report.append(f"#### {bubble['name']} {status}")
            report.append(f"- **Methods:** {len(bubble['methods'])}")
            report.append(f"- **Placeholders:** {placeholder_count}")
            report.append(f"- **Empty Methods:** {empty_count}")
            report.append(f"- **Issues:** {issue_count}")
            if bubble['issues']:
                for issue in bubble['issues'][:3]:
                    report.append(f"  - {issue['severity'].upper()}: {issue['message']}")
            report.append("")

        return "\n".join(report)

    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == '__main__':
    bubbles_dir = r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLab\packages\bubble-core\src\bubbles'

    analyzer = BubbleAnalyzer(bubbles_dir)
    analyzer.analyze()

    report = analyzer.generate_report()

    # Save report
    report_path = Path(bubbles_dir).parent.parent.parent.parent / 'docs' / 'BUBBLELAB_GAP_ANALYSIS.md'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Gap analysis report generated: {report_path}")
    print(f"Total bubbles analyzed: {analyzer.stats['total_bubbles']}")
    print(f"Total methods: {analyzer.stats['total_methods']}")
    print(f"Implementation rate: {(analyzer.stats['implemented_methods']/analyzer.stats['total_methods']*100):.1f}%")
