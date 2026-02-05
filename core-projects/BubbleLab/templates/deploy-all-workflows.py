#!/usr/bin/env python3
"""
BubbleLab Workflow Deployment Script
Deploy all production workflows to BubbleLab
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import requests
import json

class BubbleLabWorkflowDeployer:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

    def list_workflows(self) -> List[Dict]:
        """List all existing workflows"""
        response = self.session.get(f"{self.base_url}/bubble-flow")
        response.raise_for_status()
        return response.json().get('bubbleFlows', [])

    def create_workflow(self, name: str, code: str, description: str, event_type: str = 'webhook/http') -> Dict:
        """Create a new workflow"""
        response = self.session.post(
            f"{self.base_url}/bubble-flow",
            json={
                'name': name,
                'description': description,
                'eventType': event_type,
                'code': code
            }
        )
        response.raise_for_status()
        return response.json()

    def update_workflow(self, flow_id: int, code: str) -> Dict:
        """Update an existing workflow"""
        response = self.session.put(
            f"{self.base_url}/bubble-flow/{flow_id}",
            json={'code': code}
        )
        response.raise_for_status()
        return response.json()

    def activate_workflow(self, flow_id: int) -> None:
        """Activate a workflow"""
        response = self.session.post(f"{self.base_url}/bubble-flow/{flow_id}/activate")
        response.raise_for_status()

    def deploy_from_directory(self, templates_dir: str, activate: bool = True) -> Dict:
        """Deploy all workflows from directory"""
        templates_path = Path(templates_dir)
        results = {
            'deployed': [],
            'updated': [],
            'failed': [],
            'skipped': []
        }

        # Find all TypeScript files
        workflow_files = list(templates_path.rglob('*.ts'))

        print(f"Found {len(workflow_files)} workflow files")

        # Get existing workflows
        existing_workflows = {w['name']: w for w in self.list_workflows()}

        for workflow_file in workflow_files:
            # Extract workflow name from file
            relative_path = workflow_file.relative_to(templates_path)
            category = relative_path.parent.name
            file_name = workflow_file.stem

            # Create workflow name
            workflow_name = file_name.replace('-', ' ').replace('_', ' ').title()
            full_name = f"{category.title()}: {workflow_name}"

            print(f"\n{'='*60}")
            print(f"Processing: {full_name}")
            print(f"File: {workflow_file}")

            try:
                # Read workflow code
                with open(workflow_file, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Extract description from file
                description = self.extract_description(code)

                # Extract event type from code
                event_type = self.extract_event_type(code)

                # Check if workflow exists
                if full_name in existing_workflows:
                    existing_flow = existing_workflows[full_name]
                    print(f"  Updating existing workflow (ID: {existing_flow['id']})")

                    # Update workflow
                    self.update_workflow(existing_flow['id'], code)
                    results['updated'].append(full_name)
                    print(f"  [OK] Updated")
                else:
                    print(f"  Creating new workflow")

                    # Create workflow
                    flow = self.create_workflow(
                        name=full_name,
                        code=code,
                        description=description,
                        event_type=event_type
                    )

                    results['deployed'].append(full_name)
                    print(f"  [OK] Created (ID: {flow['id']})")

                    # Activate if requested
                    if activate:
                        self.activate_workflow(flow['id'])
                        print(f"  ▶️ Activated")

            except Exception as e:
                print(f"  [FAIL] Failed: {e}")
                results['failed'].append({
                    'workflow': full_name,
                    'error': str(e)
                })

        return results

    def extract_description(self, code: str) -> str:
        """Extract description from workflow code"""
        import re
        match = re.search(r'description = \'[^\']*\'', code)
        if match:
            return match.group(0).split('=')[1].strip().strip("'")
        return 'Auto-generated workflow'

    def extract_event_type(self, code: str) -> str:
        """Extract event type from workflow code"""
        import re

        # Check for schedule/cron
        if 'cronSchedule' in code or 'schedule/cron' in code:
            return 'schedule/cron'

        # Check for slack events
        if 'slack/bot_mentioned' in code:
            return 'slack/bot_mentioned'

        # Default to webhook/http
        return 'webhook/http'

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Deploy BubbleLab workflows')
    parser.add_argument('--url', default=os.getenv('BUBBLELAB_URL', 'http://localhost:3001'),
                       help='BubbleLab API URL')
    parser.add_argument('--api-key', required=True, help='BubbleLab API key (or set BUBBLELAB_API_KEY env var)')
    parser.add_argument('--templates-dir', default='./BubbleLab/templates',
                       help='Templates directory path')
    parser.add_argument('--no-activate', action='store_true',
                       help='Do not activate workflows after deployment')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deployed without actually deploying')

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('BUBBLELAB_API_KEY')
    if not api_key:
        print("[FAIL] Error: BUBBLELAB_API_KEY must be provided via --api-key or environment variable")
        sys.exit(1)

    print("="*60)
    print("BubbleLab Workflow Deployment")
    print("="*60)
    print(f"URL: {args.url}")
    print(f"Templates: {args.templates_dir}")
    print(f"Activate: {not args.no_activate}")
    print(f"Dry Run: {args.dry_run}")
    print("="*60)

    if args.dry_run:
        print("\n🔍 Dry run mode - showing what would be deployed:\n")
        templates_path = Path(args.templates_dir)
        workflow_files = list(templates_path.rglob('*.ts'))

        for workflow_file in workflow_files:
            relative_path = workflow_file.relative_to(templates_path)
            category = relative_path.parent.name
            file_name = workflow_file.stem
            workflow_name = f"{category.title()}: {file_name.replace('-', ' ').replace('_', ' ').title()}"
            print(f"  * {workflow_name}")

        print(f"\nTotal: {len(workflow_files)} workflows")
        return

    # Deploy workflows
    deployer = BubbleLabWorkflowDeployer(args.url, api_key)
    results = deployer.deploy_from_directory(
        args.templates_dir,
        activate=not args.no_activate
    )

    # Print summary
    print("\n" + "="*60)
    print("Deployment Summary")
    print("="*60)
    print(f"[OK] Created: {len(results['deployed'])}")
    print(f"🔄 Updated: {len(results['updated'])}")
    print(f"[FAIL] Failed: {len(results['failed'])}")

    if results['deployed']:
        print(f"\nCreated Workflows:")
        for name in results['deployed']:
            print(f"  [OK] {name}")

    if results['updated']:
        print(f"\nUpdated Workflows:")
        for name in results['updated']:
            print(f"  🔄 {name}")

    if results['failed']:
        print(f"\nFailed Workflows:")
        for failure in results['failed']:
            print(f"  [FAIL] {failure['workflow']}: {failure['error']}")

    print("\n" + "="*60)

    # Exit with error code if any failures
    if results['failed']:
        sys.exit(1)

if __name__ == '__main__':
    main()
