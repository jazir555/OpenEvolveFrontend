#!/usr/bin/env python3
"""
BubbleLab Comprehensive Gap Analysis Script
Analyzes all bubbles to identify implementation gaps with proper method detection
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

class BubbleGapAnalyzer:
    def __init__(self, bubbles_dir: str):
        self.bubbles_dir = Path(bubbles_dir)
        self.bubbles = {
            'service': [],
            'tool': [],
            'workflow': []
        }
        self.stats = {
            'total_bubbles': 0,
            'with_placeholders': 0,
            'incomplete_implementations': 0,
            'missing_error_handling': 0,
            'missing_timeouts': 0,
            'missing_validations': 0
        }
        self.gaps = {
            'critical': [],   # Methods that don't work at all
            'high': [],       # Incomplete implementations, missing critical features
            'medium': [],     # Missing error handling, retries, timeouts
            'low': []         # Missing logging, telemetry
        }

    def get_bubble_files(self) -> Dict[str, List[Path]]:
        """Get all main bubble implementation files"""
        files = {'service': [], 'tool': [], 'workflow': []}

        # Service bubbles
        service_dir = self.bubbles_dir / 'service-bubble'
        if service_dir.exists():
            for f in service_dir.glob('*.ts'):
                if not any(x in f.name for x in ['test', 'spec']):
                    files['service'].append(f)
            # Check subdirectories (like apify, google-sheets, notion)
            for subdir in service_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    for f in subdir.glob('*.ts'):
                        # Main implementation files only
                        if subdir.name in f.name and not any(x in f.name for x in ['test', 'spec', 'schema', 'index', 'utils']):
                            files['service'].append(f)

        # Tool bubbles
        tool_dir = self.bubbles_dir / 'tool-bubble'
        if tool_dir.exists():
            for f in tool_dir.glob('*.ts'):
                if not any(x in f.name for x in ['test', 'spec', 'template']):
                    files['tool'].append(f)

        # Workflow bubbles
        workflow_dir = self.bubbles_dir / 'workflow-bubble'
        if workflow_dir.exists():
            for f in workflow_dir.glob('*.workflow.ts'):
                files['workflow'].append(f)
            for f in workflow_dir.glob('*-agent.ts'):
                files['workflow'].append(f)

        return files

    def analyze_bubble_file(self, file_path: Path, bubble_type: str) -> Dict[str, Any]:
        """Analyze a single bubble file for gaps"""
        try:
            content = file_path.read_text(encoding='utf-8')
        except:
            return None

        bubble_name = file_path.stem.replace('-bubble', '').replace('.workflow', '')

        # Find all exported functions/methods
        methods = self.extract_methods(content)

        # Analyze each method
        method_gaps = []
        for method in methods:
            gaps = self.analyze_method(method, content, bubble_name)
            method_gaps.extend(gaps)

        # Check file-level issues
        file_issues = self.check_file_level_issues(content, bubble_name)

        return {
            'name': bubble_name,
            'type': bubble_type,
            'file_path': str(file_path),
            'methods': methods,
            'method_count': len(methods),
            'gaps': method_gaps,
            'file_issues': file_issues,
            'line_count': len(content.split('\n')),
            'has_placeholders': any(g['severity'] == 'critical' for g in method_gaps),
            'is_complete': len(method_gaps) == 0
        }

    def extract_methods(self, content: str) -> List[Dict[str, Any]]:
        """Extract actual method definitions (not array methods)"""
        methods = []

        # Match exported functions and methods
        # Pattern: export function/method, or direct function definitions
        patterns = [
            r'export\s+async\s+function\s+(\w+)\s*\(',
            r'export\s+function\s+(\w+)\s*\(',
            r'async\s+function\s+(\w+)\s*\([^)]*\)\s*{',
            r'function\s+(\w+)\s*\([^)]*\)\s*{',
            r'(\w+)\s*:\s*async\s+function',
            r'(\w+)\s*:\s*function',
            r'(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*{[^}]*export',  # Object methods with export
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                method_name = match.group(1)

                # Skip common non-method patterns
                if method_name in ['if', 'while', 'for', 'catch', 'switch', 'case']:
                    continue
                if method_name.startswith('on') or method_name.startswith('use'):
                    continue
                # Skip array methods that might be matched
                if method_name in ['map', 'filter', 'reduce', 'forEach', 'find', 'some', 'every', 'sort']:
                    # Only include if it's actually defined as a function
                    if not re.search(rf'function\s+{method_name}\s*\(', content):
                        continue

                # Extract method body
                start_pos = match.start()
                body = self.extract_method_body(content, start_pos)

                if body and len(body.strip()) > 10:
                    methods.append({
                        'name': method_name,
                        'body': body,
                        'start_pos': start_pos
                    })

        # Remove duplicates
        seen = set()
        unique_methods = []
        for m in methods:
            key = f"{m['name']}_{m['start_pos']}"
            if key not in seen:
                seen.add(key)
                unique_methods.append(m)

        return unique_methods

    def extract_method_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a method"""
        # Find opening brace
        brace_pos = content.find('{', start_pos)
        if brace_pos == -1:
            return None

        # Find matching closing brace
        depth = 1
        i = brace_pos + 1
        while i < len(content) and depth > 0:
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
            i += 1

        if depth == 0:
            return content[brace_pos:i]
        return None

    def analyze_method(self, method: Dict, content: str, bubble_name: str) -> List[Dict]:
        """Analyze a method for gaps"""
        gaps = []
        body = method['body']
        method_name = method['name']

        # Check for placeholder implementations
        if self.is_placeholder(body):
            gaps.append({
                'severity': 'critical',
                'type': 'placeholder_implementation',
                'method': method_name,
                'message': f'Method "{method_name}" contains placeholder code (TODO/FIXME/NotImplemented)'
            })

        # Check for empty implementation
        if self.is_empty(body):
            gaps.append({
                'severity': 'critical',
                'type': 'empty_implementation',
                'method': method_name,
                'message': f'Method "{method_name}" is empty or only has return statement'
            })

        # Check for error handling
        has_try_catch = 'try' in body and 'catch' in body
        has_error_param = 'error' in body or 'err' in body or 'e)' in body
        if not has_try_catch and not has_error_param and 'async' in body[:100]:
            gaps.append({
                'severity': 'medium',
                'type': 'missing_error_handling',
                'method': method_name,
                'message': f'Method "{method_name}" lacks error handling'
            })

        # Check for timeout
        has_timeout = 'timeout' in body.lower()
        has_api_call = re.search(r'(fetch|axios|request|http\.call)\(', body)
        if has_api_call and not has_timeout:
            gaps.append({
                'severity': 'medium',
                'type': 'missing_timeout',
                'method': method_name,
                'message': f'Method "{method_name}" has API calls without timeout'
            })

        # Check for retry logic
        has_retry = 'retry' in body.lower() or 'retry(' in body
        if has_api_call and not has_retry:
            gaps.append({
                'severity': 'low',
                'type': 'missing_retry',
                'method': method_name,
                'message': f'Method "{method_name}" lacks retry logic for API calls'
            })

        # Check for validation
        has_validation = re.search(r'(validate|check|verify|schema|zod)\(', body.lower())
        has_params = re.search(r'(params|input|args|options)\s*[=:]', body)
        if has_params and not has_validation:
            gaps.append({
                'severity': 'high',
                'type': 'missing_validation',
                'method': method_name,
                'message': f'Method "{method_name}" missing input validation'
            })

        return gaps

    def is_placeholder(self, body: str) -> bool:
        """Check if body contains placeholder code"""
        indicators = [
            r'TODO',
            r'FIXME',
            r'XXX',
            r'NotImplemented',
            r'throw new Error\(.*[Nn]ot implemented',
            r'\/\/\s*implement',
            r'\/\*\s*implement',
            r'placeholder'
        ]
        body_lower = body.lower()
        return any(re.search(ind, body, re.IGNORECASE) for ind in indicators)

    def is_empty(self, body: str) -> bool:
        """Check if body is effectively empty"""
        # Remove whitespace
        cleaned = re.sub(r'\s+', '', body)
        # Check for empty returns
        empty_patterns = [
            r'{}',
            r'{return;}',
            r'{returnnull;}',
            r'{returnundefined;}',
            r'{return"";}',
            r'{return\[\];}'
        ]
        return any(re.match(pattern, cleaned) for pattern in empty_patterns) or len(cleaned) < 20

    def check_file_level_issues(self, content: str, bubble_name: str) -> List[Dict]:
        """Check for file-level issues"""
        issues = []

        # Check for hardcoded credentials
        if re.search(r'(api_key|apikey|password|secret|token)\s*=\s*["\'][^"\']+["\']', content):
            issues.append({
                'severity': 'critical',
                'type': 'hardcoded_credentials',
                'message': 'Potentially hardcoded credentials detected'
            })

        # Check for missing imports
        if 'credential' in content.lower() and 'credential' not in content[:500]:
            issues.append({
                'severity': 'high',
                'type': 'credential_handling',
                'message': 'Uses credentials but may not import credential manager'
            })

        # Check for logging
        has_logging = re.search(r'(logger|log\.\s*|console\.(log|error|warn))', content)
        if not has_logging:
            issues.append({
                'severity': 'low',
                'type': 'missing_logging',
                'message': 'No structured logging found'
            })

        return issues

    def analyze_all(self):
        """Analyze all bubbles"""
        files = self.get_bubble_files()

        for bubble_type, file_list in files.items():
            for file_path in file_list:
                bubble = self.analyze_bubble_file(file_path, bubble_type)
                if bubble and bubble['method_count'] > 0:
                    self.bubbles[bubble_type].append(bubble)

        self.calculate_stats()
        self.categorize_gaps()

    def calculate_stats(self):
        """Calculate statistics"""
        all_bubbles = self.bubbles['service'] + self.bubbles['tool'] + self.bubbles['workflow']
        self.stats['total_bubbles'] = len(all_bubbles)
        self.stats['with_placeholders'] = sum(1 for b in all_bubbles if b['has_placeholders'])
        self.stats['incomplete_implementations'] = sum(1 for b in all_bubbles if not b['is_complete'])

        for bubble in all_bubbles:
            for gap in bubble['gaps']:
                if gap['type'] == 'missing_error_handling':
                    self.stats['missing_error_handling'] += 1
                elif gap['type'] == 'missing_timeout':
                    self.stats['missing_timeouts'] += 1
                elif gap['type'] == 'missing_validation':
                    self.stats['missing_validations'] += 1

    def categorize_gaps(self):
        """Categorize all gaps by severity"""
        all_bubbles = self.bubbles['service'] + self.bubbles['tool'] + self.bubbles['workflow']

        for bubble in all_bubbles:
            # Add method gaps
            for gap in bubble['gaps']:
                self.gaps[gap['severity']].append({
                    'bubble': bubble['name'],
                    'type': bubble['type'],
                    'method': gap['method'],
                    'gap_type': gap['type'],
                    'message': gap['message'],
                    'file': bubble['file_path']
                })

            # Add file-level gaps
            for issue in bubble['file_issues']:
                self.gaps[issue['severity']].append({
                    'bubble': bubble['name'],
                    'type': bubble['type'],
                    'method': 'N/A',
                    'gap_type': issue['type'],
                    'message': issue['message'],
                    'file': bubble['file_path']
                })

    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        lines = []
        lines.append("# BubbleLab Comprehensive Gap Analysis Report")
        lines.append("")
        lines.append(f"**Generated:** {self.get_timestamp()}")
        lines.append(f"**Analysis Scope:** All 68+ bubbles in the BubbleLab system")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        complete = sum(1 for b in self.bubbles['service'] + self.bubbles['tool'] + self.bubbles['workflow'] if b['is_complete'])
        total = self.stats['total_bubbles']
        completion_rate = (complete / total * 100) if total > 0 else 0

        lines.append(f"### Overview")
        lines.append("")
        lines.append(f"- **Total Bubbles Analyzed:** {total}")
        lines.append(f"- **Service Bubbles:** {len(self.bubbles['service'])}")
        lines.append(f"- **Tool Bubbles:** {len(self.bubbles['tool'])}")
        lines.append(f"- **Workflow Bubbles:** {len(self.bubbles['workflow'])}")
        lines.append("")
        lines.append(f"- **Fully Complete:** {complete} ({completion_rate:.1f}%)")
        lines.append(f"- **With Placeholders:** {self.stats['with_placeholders']}")
        lines.append(f"- **Incomplete:** {self.stats['incomplete_implementations']}")
        lines.append("")

        # Critical Issues
        lines.append("### Critical Issues Summary")
        lines.append("")
        lines.append(f"- **Bubbles with Placeholder Code:** {len([g for g in self.gaps['critical'] if g['gap_type'] == 'placeholder_implementation'])}")
        lines.append(f"- **Empty Implementations:** {len([g for g in self.gaps['critical'] if g['gap_type'] == 'empty_implementation'])}")
        lines.append(f"- **Hardcoded Credentials:** {len([g for g in self.gaps['critical'] if g['gap_type'] == 'hardcoded_credentials'])}")
        lines.append("")

        # High Priority Issues
        lines.append("### High Priority Issues Summary")
        lines.append("")
        lines.append(f"- **Missing Input Validation:** {self.stats['missing_validations']}")
        lines.append("")

        # Medium Priority Issues
        lines.append("### Medium Priority Issues Summary")
        lines.append("")
        lines.append(f"- **Missing Error Handling:** {self.stats['missing_error_handling']}")
        lines.append(f"- **Missing Timeouts:** {self.stats['missing_timeouts']}")
        lines.append("")

        # Detailed Critical Gaps
        lines.append("## 1. Critical Gaps (Must Fix)")
        lines.append("")
        lines.append("These gaps will cause failures in production and must be fixed immediately.")
        lines.append("")

        if self.gaps['critical']:
            # Group by bubble
            by_bubble = defaultdict(list)
            for gap in self.gaps['critical']:
                by_bubble[gap['bubble']].append(gap)

            for bubble_name in sorted(by_bubble.keys()):
                lines.append(f"### {bubble_name}")
                for gap in by_bubble[bubble_name]:
                    lines.append(f"- **{gap['gap_type'].replace('_', ' ').title()}:** {gap['message']}")
                    if gap['method'] != 'N/A':
                        lines.append(f"  - Method: `{gap['method']}`")
                    lines.append(f"  - File: `{Path(gap['file']).name}`")
                lines.append("")
        else:
            lines.append("No critical gaps found!")
            lines.append("")

        # High Priority Gaps
        lines.append("## 2. High Priority Gaps")
        lines.append("")
        lines.append("These gaps impact reliability and should be fixed soon.")
        lines.append("")

        if self.gaps['high']:
            # Group by type
            by_type = defaultdict(list)
            for gap in self.gaps['high']:
                by_type[gap['gap_type']].append(gap)

            for gap_type, gaps in sorted(by_type.items()):
                lines.append(f"### {gap_type.replace('_', ' ').title()} ({len(gaps)} occurrences)")
                lines.append("")
                for gap in gaps[:10]:
                    lines.append(f"- **{gap['bubble']}:** {gap['message']}")
                    lines.append(f"  - Method: `{gap['method']}`")
                if len(gaps) > 10:
                    lines.append(f"- ... and {len(gaps) - 10} more")
                lines.append("")
        else:
            lines.append("No high priority gaps found!")
            lines.append("")

        # Medium Priority Gaps
        lines.append("## 3. Medium Priority Gaps")
        lines.append("")
        if self.gaps['medium']:
            lines.append(f"**Total Issues:** {len(self.gaps['medium'])}")
            lines.append("")

            by_type = defaultdict(list)
            for gap in self.gaps['medium']:
                by_type[gap['gap_type']].append(gap)

            for gap_type, gaps in sorted(by_type.items()):
                lines.append(f"### {gap_type.replace('_', ' ').title()}")
                lines.append(f"**Affected Bubbles:** {len(gaps)}")
                lines.append("")
                # List top 5 affected bubbles
                bubble_counts = defaultdict(int)
                for g in gaps:
                    bubble_counts[g['bubble']] += 1
                for bubble, count in sorted(bubble_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    lines.append(f"- {bubble}: {count} methods")
                lines.append("")
        else:
            lines.append("No medium priority gaps found!")
            lines.append("")

        # Low Priority Gaps
        lines.append("## 4. Low Priority Gaps (Enhancements)")
        lines.append("")
        lines.append(f"**Total Issues:** {len(self.gaps['low'])}")
        lines.append("")

        # Implementation Roadmap
        lines.append("## 5. Implementation Roadmap")
        lines.append("")

        lines.append("### Phase 1: Critical Fixes (Week 1) - MUST COMPLETE")
        lines.append("**Priority:** P0 - Blocking production")
        lines.append("")
        lines.append("1. **Remove Placeholder Code**")
        placeholder_count = len([g for g in self.gaps['critical'] if g['gap_type'] == 'placeholder_implementation'])
        lines.append(f"   - Bubbles affected: {placeholder_count}")
        lines.append(f"   - Estimated effort: {placeholder_count * 4} hours")
        lines.append("   - Action: Replace all TODO/FIXME with actual implementations")
        lines.append("")

        lines.append("2. **Implement Empty Methods**")
        empty_count = len([g for g in self.gaps['critical'] if g['gap_type'] == 'empty_implementation'])
        lines.append(f"   - Methods affected: {empty_count}")
        lines.append(f"   - Estimated effort: {empty_count * 2} hours")
        lines.append("   - Action: Add proper implementations")
        lines.append("")

        lines.append("3. **Remove Hardcoded Credentials**")
        creds_count = len([g for g in self.gaps['critical'] if g['gap_type'] == 'hardcoded_credentials'])
        lines.append(f"   - Files affected: {creds_count}")
        lines.append(f"   - Estimated effort: {creds_count * 1} hours")
        lines.append("   - Action: Move to environment variables or credential service")
        lines.append("")

        lines.append("### Phase 2: High Priority Fixes (Week 2)")
        lines.append("**Priority:** P1 - Important for reliability")
        lines.append("")

        lines.append("1. **Add Input Validation**")
        lines.append(f"   - Methods affected: {self.stats['missing_validations']}")
        lines.append(f"   - Estimated effort: {self.stats['missing_validations'] * 0.5} hours")
        lines.append("   - Action: Add Zod schemas or validation checks")
        lines.append("")

        lines.append("2. **Add Timeouts to API Calls**")
        lines.append(f"   - Methods affected: {self.stats['missing_timeouts']}")
        lines.append(f"   - Estimated effort: {self.stats['missing_timeouts'] * 0.25} hours")
        lines.append("   - Action: Add timeout parameter to fetch/axios calls")
        lines.append("")

        lines.append("### Phase 3: Medium Priority Fixes (Week 3)")
        lines.append("**Priority:** P2 - Production readiness")
        lines.append("")

        lines.append("1. **Add Error Handling**")
        lines.append(f"   - Methods affected: {self.stats['missing_error_handling']}")
        lines.append(f"   - Estimated effort: {self.stats['missing_error_handling'] * 0.5} hours")
        lines.append("   - Action: Wrap async operations in try-catch blocks")
        lines.append("")

        lines.append("2. **Add Retry Logic**")
        lines.append(f"   - API calls affected: ~{len(self.gaps['low'])}")
        lines.append(f"   - Estimated effort: {len(self.gaps['low']) * 0.5} hours")
        lines.append("   - Action: Implement exponential backoff retry")
        lines.append("")

        # Detailed Bubble Status
        lines.append("## 6. Detailed Bubble Status")
        lines.append("")

        # Service Bubbles
        lines.append("### 6.1 Service Bubbles")
        lines.append("")
        lines.append("| Bubble | Methods | Status | Issues |")
        lines.append("|--------|---------|--------|--------|")
        for bubble in sorted(self.bubbles['service'], key=lambda x: x['name']):
            status = "✅ Complete" if bubble['is_complete'] else "⚠️ Incomplete"
            issues = len(bubble['gaps'])
            lines.append(f"| {bubble['name']} | {bubble['method_count']} | {status} | {issues} |")
        lines.append("")

        # Tool Bubbles
        lines.append("### 6.2 Tool Bubbles")
        lines.append("")
        lines.append("| Bubble | Methods | Status | Issues |")
        lines.append("|--------|---------|--------|--------|")
        for bubble in sorted(self.bubbles['tool'], key=lambda x: x['name']):
            status = "✅ Complete" if bubble['is_complete'] else "⚠️ Incomplete"
            issues = len(bubble['gaps'])
            lines.append(f"| {bubble['name']} | {bubble['method_count']} | {status} | {issues} |")
        lines.append("")

        # Workflow Bubbles
        lines.append("### 6.3 Workflow Bubbles")
        lines.append("")
        lines.append("| Bubble | Methods | Status | Issues |")
        lines.append("|--------|---------|--------|--------|")
        for bubble in sorted(self.bubbles['workflow'], key=lambda x: x['name']):
            status = "✅ Complete" if bubble['is_complete'] else "⚠️ Incomplete"
            issues = len(bubble['gaps'])
            lines.append(f"| {bubble['name']} | {bubble['method_count']} | {status} | {issues} |")
        lines.append("")

        # Conclusion
        lines.append("## 7. Recommendations")
        lines.append("")
        lines.append("### Immediate Actions")
        lines.append("")
        lines.append("1. **Prioritize Critical Fixes:** Start with placeholder and empty implementations")
        lines.append("2. **Security Review:** Remove any hardcoded credentials immediately")
        lines.append("3. **Testing:** Add integration tests for all critical paths")
        lines.append("")

        lines.append("### Best Practices")
        lines.append("")
        lines.append("1. **Error Handling:** All async methods should have try-catch blocks")
        lines.append("2. **Timeouts:** All API calls should have explicit timeouts (5-30 seconds)")
        lines.append("3. **Validation:** Validate all inputs using Zod or similar")
        lines.append("4. **Logging:** Use structured logging with correlation IDs")
        lines.append("5. **Credentials:** Use the credential service, never hardcode")
        lines.append("")

        lines.append("### Estimated Total Effort")
        lines.append("")
        total_hours = (
            (len([g for g in self.gaps['critical'] if g['gap_type'] == 'placeholder_implementation']) * 4) +
            (len([g for g in self.gaps['critical'] if g['gap_type'] == 'empty_implementation']) * 2) +
            (self.stats['missing_validations'] * 0.5) +
            (self.stats['missing_timeouts'] * 0.25) +
            (self.stats['missing_error_handling'] * 0.5)
        )
        lines.append(f"- **Phase 1 (Critical):** {total_hours * 0.4:.1f} hours")
        lines.append(f"- **Phase 2 (High):** {total_hours * 0.3:.1f} hours")
        lines.append(f"- **Phase 3 (Medium):** {total_hours * 0.3:.1f} hours")
        lines.append(f"- **Total:** {total_hours:.1f} hours (~{total_hours/8:.1f} days)")
        lines.append("")

        return "\n".join(lines)

    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == '__main__':
    bubbles_dir = r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLab\packages\bubble-core\src\bubbles'

    print("Analyzing BubbleLab bubbles...")
    analyzer = BubbleGapAnalyzer(bubbles_dir)
    analyzer.analyze_all()

    print(f"Found {analyzer.stats['total_bubbles']} bubbles")
    print(f"Service bubbles: {len(analyzer.bubbles['service'])}")
    print(f"Tool bubbles: {len(analyzer.bubbles['tool'])}")
    print(f"Workflow bubbles: {len(analyzer.bubbles['workflow'])}")

    report = analyzer.generate_report()

    # Save report
    report_path = Path(bubbles_dir).parent.parent.parent.parent / 'docs' / 'BUBBLELAB_COMPREHENSIVE_GAP_ANALYSIS.md'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(f"\nCritical gaps: {len(analyzer.gaps['critical'])}")
    print(f"High priority gaps: {len(analyzer.gaps['high'])}")
    print(f"Medium priority gaps: {len(analyzer.gaps['medium'])}")
    print(f"Low priority gaps: {len(analyzer.gaps['low'])}")
