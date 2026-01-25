#!/usr/bin/env python3
"""
Comprehensive Authentication Testing Suite
Tests webhook.ts, BubbleSidePanel.tsx, and subscription.ts authentication fixes
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}[PASS] {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")

class AuthTester:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.bubblelab_dir = self.root_dir / "BubbleLab"
        self.test_results: List[Tuple[str, bool, str]] = []

    def run_all_tests(self) -> bool:
        """Run all authentication tests"""
        print_header("COMPREHENSIVE AUTHENTICATION TESTING SUITE")

        all_passed = True

        # Test 1: webhook.ts authentication
        all_passed &= self.test_webhook_auth()

        # Test 2: BubbleSidePanel.tsx authentication
        all_passed &= self.test_bubble_sidepanel_auth()

        # Test 3: subscription.ts authentication
        all_passed &= self.test_subscription_auth()

        # Test 4: Auth middleware verification
        all_passed &= self.test_auth_middleware()

        # Test 5: Clerk configuration
        all_passed &= self.test_clerk_config()

        # Test 6: Hard-coded value detection
        all_passed &= self.test_no_hardcoded_values()

        # Print summary
        self.print_summary()

        return all_passed

    def test_webhook_auth(self) -> bool:
        """Test 1: Verify webhook.ts authentication"""
        print_header("TEST 1: webhook.ts Authentication")
        passed = True

        webhook_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "utils" / "webhook.ts"

        if not webhook_file.exists():
            print_error(f"webhook.ts not found at {webhook_file}")
            return False

        content = webhook_file.read_text()

        # Test 1.1: Check getWebhookUrl function signature
        print_info("Test 1.1: Checking getWebhookUrl function signature...")
        if 'getWebhookUrl(userId: string' in content:
            print_success("getWebhookUrl accepts userId parameter")
        else:
            print_error("getWebhookUrl does not accept userId parameter")
            passed = False

        # Test 1.2: Verify userId is used in URL construction
        print_info("Test 1.2: Verifying userId is used in URL construction...")
        if '${userId}' in content or '/webhook/${userId}/' in content:
            print_success("userId is properly used in webhook URL")
        else:
            print_error("userId is not used in webhook URL construction")
            passed = False

        # Test 1.3: Check for hard-coded "1"
        print_info("Test 1.3: Checking for hard-coded user IDs...")
        # Look for various hard-coded patterns
        hardcoded_patterns = [
            r'/webhook/1/',
            r'userId\s*=\s*["\']1["\']',
            r'userId\s*=\s*1\b',
            r'getWebhookUrl\(["\']1["\']',
        ]

        found_hardcoded = False
        for pattern in hardcoded_patterns:
            if re.search(pattern, content):
                print_error(f"Found hard-coded user ID pattern: {pattern}")
                found_hardcoded = True
                passed = False

        if not found_hardcoded:
            print_success("No hard-coded user IDs found")

        # Test 1.4: Verify function extracts userId from request context
        print_info("Test 1.4: Verifying userId extraction documentation...")
        if 'auth middleware' in content.lower() or 'request context' in content.lower():
            print_success("Documentation mentions userId extraction from auth middleware")
        else:
            print_warning("Should document userId extraction from auth middleware")

        self.test_results.append(("webhook.ts Authentication", passed, "User ID extraction and usage"))
        return passed

    def test_bubble_sidepanel_auth(self) -> bool:
        """Test 2: Verify BubbleSidePanel.tsx authentication"""
        print_header("TEST 2: BubbleSidePanel.tsx Authentication")
        passed = True

        sidepanel_file = self.bubblelab_dir / "apps" / "bubble-studio" / "src" / "components" / "BubbleSidePanel.tsx"

        if not sidepanel_file.exists():
            print_error(f"BubbleSidePanel.tsx not found at {sidepanel_file}")
            return False

        content = sidepanel_file.read_text()

        # Test 2.1: Check useUser hook import
        print_info("Test 2.1: Checking useUser hook import...")
        if "from '../hooks/useUser'" in content or 'useUser' in content:
            print_success("useUser hook is imported")
        else:
            print_error("useUser hook is not imported")
            passed = False

        # Test 2.2: Verify useUser is called
        print_info("Test 2.2: Verifying useUser hook is used...")
        if 'const { user, isLoaded: isUserLoaded } = useUser()' in content:
            print_success("useUser hook is properly destructured")
        else:
            print_error("useUser hook is not properly called")
            passed = False

        # Test 2.3: Check getUserName function
        print_info("Test 2.3: Checking getUserName function implementation...")
        if 'const getUserName = () =>' in content or 'function getUserName' in content:
            print_success("getUserName function exists")

            # Check if it handles loading state
            if 'isUserLoaded' in content and 'Loading' in content:
                print_success("getUserName handles loading state")
            else:
                print_warning("getUserName should handle loading state")

            # Check if it handles unauthenticated state
            if 'Guest' in content:
                print_success("getUserName handles unauthenticated state (Guest)")
            else:
                print_warning("getUserName should handle unauthenticated state")

            # Check if it uses real user data
            if 'user.fullName' in content or 'user.firstName' in content or 'user.emailAddresses' in content:
                print_success("getUserName uses real user data")
            else:
                print_error("getUserName does not use real user data")
                passed = False
        else:
            print_error("getUserName function not found")
            passed = False

        # Test 2.4: Check for hard-coded 'User' string
        print_info("Test 2.4: Checking for hard-coded 'User' string...")
        # Look for hard-coded 'User' in display contexts
        hardcoded_user_pattern = r"userName\s*[:=]\s*['\"]User['\"]"
        if re.search(hardcoded_user_pattern, content):
            print_error("Found hard-coded 'User' string for userName")
            passed = False
        else:
            print_success("No hard-coded 'User' string found")

        # Test 2.5: Verify userName is passed to MilkTea
        print_info("Test 2.5: Verifying userName is passed to API...")
        if 'userName: getUserName()' in content:
            print_success("userName is passed to MilkTea mutation")
        else:
            print_warning("userName should be passed to MilkTea mutation")

        self.test_results.append(("BubbleSidePanel.tsx Authentication", passed, "useUser hook and user name display"))
        return passed

    def test_subscription_auth(self) -> bool:
        """Test 3: Verify subscription.ts authentication"""
        print_header("TEST 3: subscription.ts Authentication")
        passed = True

        subscription_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "routes" / "subscription.ts"

        if not subscription_file.exists():
            print_error(f"subscription.ts not found at {subscription_file}")
            return False

        content = subscription_file.read_text()

        # Test 3.1: Check auth middleware usage
        print_info("Test 3.1: Checking auth middleware is applied...")
        if "authMiddleware" in content or "app.use('*'" in content:
            print_success("Auth middleware is applied to routes")
        else:
            print_error("Auth middleware is not applied")
            passed = False

        # Test 3.2: Verify getUserId is called
        print_info("Test 3.2: Verifying getUserId is called...")
        if 'getUserId(c)' in content or 'const userId = getUserId' in content:
            print_success("getUserId is called to extract user ID")
        else:
            print_error("getUserId is not called")
            passed = False

        # Test 3.3: Check subscription status determination
        print_info("Test 3.3: Checking subscription status determination...")
        if 'isActive' in content:
            # Look for the logic that determines isActive
            if 'subscriptionInfo.plan' in content or 'hackathonOffer?.isActive' in content:
                print_success("Subscription status is determined from real data")
            else:
                print_warning("Subscription status logic should use real data")
        else:
            print_error("isActive is not determined")
            passed = False

        # Test 3.4: Check for hard-coded isActive: true
        print_info("Test 3.4: Checking for hard-coded isActive: true...")
        hardcoded_active_pattern = r'isActive\s*:\s*true\s*(?![,}])'
        if re.search(hardcoded_active_pattern, content):
            # Make sure it's not part of a conditional expression
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'isActive' in line and 'true' in line:
                    # Check if it's a conditional (should be ok)
                    if '?' in line or '||' in line or '&&' in line:
                        continue
                    else:
                        print_error(f"Line {i+1}: Possible hard-coded isActive: true")
                        passed = False
                        break
            else:
                print_success("No unconditional hard-coded isActive: true found")
        else:
            print_success("No hard-coded isActive: true found")

        # Test 3.5: Verify hackathon offer logic
        print_info("Test 3.5: Checking hackathon offer logic...")
        if 'hackathonOffer' in content:
            print_success("Hackathon offer logic exists")
            if 'getActiveHackathonOfferForResponse' in content:
                print_success("Uses helper function to get active hackathon offer")
        else:
            print_warning("Hackathon offer logic should be implemented")

        # Test 3.6: Verify special offer logic
        print_info("Test 3.6: Checking special offer logic...")
        if 'specialOffer' in content:
            print_success("Special offer logic exists")
            if 'getSpecialOfferForResponse' in content:
                print_success("Uses helper function to get special offer")
        else:
            print_warning("Special offer logic should be implemented")

        self.test_results.append(("subscription.ts Authentication", passed, "Subscription status from real data"))
        return passed

    def test_auth_middleware(self) -> bool:
        """Test 4: Verify auth middleware"""
        print_header("TEST 4: Auth Middleware Verification")
        passed = True

        auth_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "middleware" / "auth.ts"

        if not auth_file.exists():
            print_error(f"auth.ts not found at {auth_file}")
            return False

        content = auth_file.read_text()

        # Test 4.1: Check JWT verification
        print_info("Test 4.1: Checking JWT verification...")
        if 'verifyToken' in content or '@clerk/backend' in content:
            print_success("JWT token is verified using Clerk backend")
        else:
            print_error("JWT verification is not implemented")
            passed = False

        # Test 4.2: Verify userId extraction
        print_info("Test 4.2: Verifying userId extraction from token...")
        if 'payload.sub' in content or "payload['sub']" in content:
            print_success("userId is extracted from JWT payload (sub)")
        else:
            print_error("userId is not extracted from JWT payload")
            passed = False

        # Test 4.3: Check userId is set in context
        print_info("Test 4.3: Checking userId is set in request context...")
        if "c.set('userId'" in content or 'c.set("userId"' in content:
            print_success("userId is set in request context")
        else:
            print_error("userId is not set in request context")
            passed = False

        # Test 4.4: Verify getUserId helper
        print_info("Test 4.4: Verifying getUserId helper function...")
        if 'export function getUserId' in content:
            print_success("getUserId helper function exists")
            if 'c.get(\'userId\')' in content or 'c.get("userId")' in content:
                print_success("getUserId extracts userId from context")
            else:
                print_error("getUserId does not extract from context")
                passed = False
        else:
            print_error("getUserId helper function not found")
            passed = False

        # Test 4.5: Check subscription info extraction
        print_info("Test 4.5: Checking subscription info extraction...")
        if 'extractSubscriptionInfoFromPayload' in content:
            print_success("Subscription info is extracted from JWT payload")
        else:
            print_warning("Subscription info extraction should be implemented")

        self.test_results.append(("Auth Middleware", passed, "JWT verification and userId extraction"))
        return passed

    def test_clerk_config(self) -> bool:
        """Test 5: Verify Clerk configuration"""
        print_header("TEST 5: Clerk Configuration")
        passed = True

        # Check useUser hook
        useuser_file = self.bubblelab_dir / "apps" / "bubble-studio" / "src" / "hooks" / "useUser.ts"

        if not useuser_file.exists():
            print_error(f"useUser.ts not found at {useuser_file}")
            return False

        content = useuser_file.read_text()

        # Test 5.1: Check Clerk import
        print_info("Test 5.1: Checking Clerk import...")
        if '@clerk/clerk-react' in content:
            print_success("Clerk React SDK is imported")
        else:
            print_error("Clerk React SDK is not imported")
            passed = False

        # Test 5.2: Verify useClerkUser usage
        print_info("Test 5.2: Verifying useClerkUser hook...")
        if 'useClerkUser' in content:
            print_success("Clerk's useUser hook is used")
        else:
            print_error("Clerk's useUser hook is not used")
            passed = False

        # Test 5.3: Check dev mode handling
        print_info("Test 5.3: Checking development mode handling...")
        if 'DISABLE_AUTH' in content:
            print_success("DISABLE_AUTH env var is checked for dev mode")
        else:
            print_warning("Should handle DISABLE_AUTH env var")

        # Test 5.4: Check mock user data for dev
        print_info("Test 5.4: Checking mock user data structure...")
        if 'mock-user-id' in content and 'Dev User' in content:
            print_success("Mock user data is provided for dev mode")
        else:
            print_warning("Mock user data should be provided for dev mode")

        # Test 5.5: Verify user object structure
        print_info("Test 5.5: Verifying user object structure...")
        required_fields = ['emailAddresses', 'firstName', 'lastName', 'fullName']
        missing_fields = []
        for field in required_fields:
            if field not in content:
                missing_fields.append(field)

        if not missing_fields:
            print_success("User object has all required fields")
        else:
            print_warning(f"User object missing fields: {', '.join(missing_fields)}")

        self.test_results.append(("Clerk Configuration", passed, "Clerk SDK integration and configuration"))
        return passed

    def test_no_hardcoded_values(self) -> bool:
        """Test 6: Comprehensive check for hard-coded values"""
        print_header("TEST 6: Hard-coded Value Detection")
        passed = True

        files_to_check = [
            (self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "utils" / "webhook.ts", "webhook.ts"),
            (self.bubblelab_dir / "apps" / "bubble-studio" / "src" / "components" / "BubbleSidePanel.tsx", "BubbleSidePanel.tsx"),
            (self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "routes" / "subscription.ts", "subscription.ts"),
        ]

        hardcoded_patterns = {
            'user_id': [
                r'userId\s*=\s*["\']1["\']',
                r'userId\s*=\s*1\b',
                r'["\']user["\']\s*:\s*["\']1["\']',
                r'/user/1/',
                r'/webhook/1/',
            ],
            'user_name': [
                r'userName\s*:\s*["\']User["\']',
                r'user["\']\s*:\s*["\']User["\']',
                r'name\s*:\s*["\']User["\']',
            ],
            'subscription': [
                r'isActive\s*:\s*true\s*[;\n]',
                r'plan\s*:\s*["\']pro["\']\s*[,;\n]',
            ]
        }

        for file_path, file_name in files_to_check:
            if not file_path.exists():
                print_warning(f"Skipping {file_name} - not found")
                continue

            print_info(f"Checking {file_name}...")
            content = file_path.read_text()
            file_passed = True

            for category, patterns in hardcoded_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = content.split('\n')[line_num - 1].strip()
                        print_error(f"{file_name}:{line_num}: Found hard-coded {category}: {line_content[:60]}...")
                        file_passed = False
                        passed = False

            if file_passed:
                print_success(f"{file_name}: No hard-coded values found")

        self.test_results.append(("Hard-coded Value Detection", passed, "No hard-coded user IDs or names"))
        return passed

    def print_summary(self):
        """Print test summary"""
        print_header("TEST SUMMARY")

        total = len(self.test_results)
        passed = sum(1 for _, p, _ in self.test_results if p)
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {failed}{Colors.END}\n")

        for test_name, test_passed, description in self.test_results:
            status = f"{Colors.GREEN}PASS{Colors.END}" if test_passed else f"{Colors.RED}FAIL{Colors.END}"
            print(f"{status} - {test_name}: {description}")

        print()
        if passed == total:
            print(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.END}")
            print(f"{Colors.GREEN}Authentication is properly implemented with real user data.{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}SOME TESTS FAILED!{Colors.END}")
            print(f"{Colors.RED}Please review the authentication implementation.{Colors.END}")

        print()

def main():
    """Main entry point"""
    root_dir = Path(__file__).parent.absolute()

    tester = AuthTester(str(root_dir))
    all_passed = tester.run_all_tests()

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
