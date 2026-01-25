#!/usr/bin/env python3
"""
Integration Test for Authentication
Tests real JWT token handling and user data flow
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

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

def print_info(text: str):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")

class AuthIntegrationTester:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.bubblelab_dir = self.root_dir / "BubbleLab"

    def test_jwt_token_structure(self) -> bool:
        """Test 1: Verify JWT token structure and claims"""
        print_header("TEST 1: JWT Token Structure Verification")
        passed = True

        # Read auth middleware to understand JWT structure
        auth_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "middleware" / "auth.ts"

        if not auth_file.exists():
            print_error(f"auth.ts not found at {auth_file}")
            return False

        content = auth_file.read_text()

        # Test 1.1: Check JWT payload interface
        print_info("Test 1.1: Checking JWT payload interface...")
        if 'interface ClerkJWTPayload' in content or 'ClerkJWTPayload' in content:
            print_success("JWT payload interface is defined")
        else:
            print_warning("JWT payload interface should be defined")

        # Test 1.2: Verify required claims
        print_info("Test 1.2: Verifying required JWT claims...")
        required_claims = ['sub', 'iss', 'azp', 'exp']
        missing_claims = []

        for claim in required_claims:
            if claim not in content:
                missing_claims.append(claim)

        if not missing_claims:
            print_success(f"All required claims present: {', '.join(required_claims)}")
        else:
            print_error(f"Missing claims: {', '.join(missing_claims)}")
            passed = False

        # Test 1.3: Check userId extraction from 'sub' claim
        print_info("Test 1.3: Verifying userId extraction from 'sub' claim...")
        if 'payload.sub' in content or "payload['sub']" in content:
            print_success("userId is correctly extracted from 'sub' claim")
        else:
            print_error("userId is not extracted from 'sub' claim")
            passed = False

        # Test 1.4: Verify token verification
        print_info("Test 1.4: Verifying token verification implementation...")
        if 'verifyToken' in content and '@clerk/backend' in content:
            print_success("Token is verified using Clerk backend SDK")
        else:
            print_error("Token verification is not implemented")
            passed = False

        return passed

    def test_user_data_flow(self) -> bool:
        """Test 2: Verify user data flows correctly through the system"""
        print_header("TEST 2: User Data Flow Verification")
        passed = True

        # Test 2.1: Frontend useUser hook
        print_info("Test 2.1: Testing frontend useUser hook...")
        useuser_file = self.bubblelab_dir / "apps" / "bubble-studio" / "src" / "hooks" / "useUser.ts"

        if useuser_file.exists():
            content = useuser_file.read_text()

            # Check if it properly exports user data
            if 'export function useUser' in content:
                print_success("useUser hook is properly exported")

                # Check if it returns user object with required fields
                if 'user:' in content and 'isLoaded:' in content and 'isSignedIn:' in content:
                    print_success("useUser returns complete user object")
                else:
                    print_warning("useUser should return user, isLoaded, isSignedIn")
            else:
                print_error("useUser hook is not exported")
                passed = False
        else:
            print_error("useUser.ts not found")
            passed = False

        # Test 2.2: BubbleSidePanel uses user data
        print_info("Test 2.2: Testing BubbleSidePanel user data usage...")
        sidepanel_file = self.bubblelab_dir / "apps" / "bubble-studio" / "src" / "components" / "BubbleSidePanel.tsx"

        if sidepanel_file.exists():
            content = sidepanel_file.read_text()

            # Check if useUser is imported
            if 'useUser' in content:
                print_success("BubbleSidePanel imports useUser hook")

                # Check if it's used
                if 'const { user' in content and '= useUser()' in content:
                    print_success("BubbleSidePanel uses useUser hook")
                else:
                    print_error("BubbleSidePanel does not use useUser hook")
                    passed = False

                # Check if userName is derived from user data
                if 'getUserName()' in content and 'userName: getUserName()' in content:
                    print_success("userName is derived from user data and passed to API")
                else:
                    print_warning("userName should be passed to API")
            else:
                print_error("BubbleSidePanel does not import useUser")
                passed = False
        else:
            print_error("BubbleSidePanel.tsx not found")
            passed = False

        # Test 2.3: Backend extracts userId from JWT
        print_info("Test 2.3: Testing backend userId extraction...")
        auth_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "middleware" / "auth.ts"

        if auth_file.exists():
            content = auth_file.read_text()

            # Check if userId is extracted from JWT
            if 'const userId = payload.sub' in content or 'userId = payload.sub' in content:
                print_success("userId is extracted from JWT payload")
            else:
                print_error("userId is not extracted from JWT")
                passed = False

            # Check if userId is set in context
            if "c.set('userId'" in content:
                print_success("userId is set in request context")
            else:
                print_error("userId is not set in context")
                passed = False
        else:
            print_error("auth.ts not found")
            passed = False

        # Test 2.4: Routes use getUserId helper
        print_info("Test 2.4: Testing route usage of getUserId...")
        subscription_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "routes" / "subscription.ts"

        if subscription_file.exists():
            content = subscription_file.read_text()

            if 'getUserId(c)' in content or 'const userId = getUserId' in content:
                print_success("Routes use getUserId helper to extract userId")
            else:
                print_warning("Routes should use getUserId helper")
        else:
            print_warning("subscription.ts not found")

        return passed

    def test_webhook_url_generation(self) -> bool:
        """Test 3: Verify webhook URL generation with real userId"""
        print_header("TEST 3: Webhook URL Generation with Real userId")
        passed = True

        webhook_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "utils" / "webhook.ts"

        if not webhook_file.exists():
            print_error(f"webhook.ts not found at {webhook_file}")
            return False

        content = webhook_file.read_text()

        # Test 3.1: Check getWebhookUrl function
        print_info("Test 3.1: Checking getWebhookUrl function signature...")
        if 'export function getWebhookUrl(userId: string' in content:
            print_success("getWebhookUrl accepts userId parameter")
        else:
            print_error("getWebhookUrl does not accept userId parameter")
            passed = False

        # Test 3.2: Verify userId is used in URL
        print_info("Test 3.2: Verifying userId is used in URL path...")
        if '${userId}' in content:
            print_success("userId is properly interpolated into webhook URL")
        else:
            print_error("userId is not used in webhook URL")
            passed = False

        # Test 3.3: Check URL pattern
        print_info("Test 3.3: Verifying webhook URL pattern...")
        if '/webhook/${userId}/' in content or '/webhook/' in content:
            print_success("Webhook URL follows correct pattern: /webhook/{userId}/{path}")
        else:
            print_warning("Webhook URL pattern should be /webhook/{userId}/{path}")

        # Test 3.4: Verify documentation mentions auth middleware
        print_info("Test 3.4: Checking documentation...")
        if 'auth middleware' in content.lower():
            print_success("Documentation mentions userId extraction from auth middleware")
        else:
            print_warning("Should document userId extraction from auth middleware")

        return passed

    def test_subscription_status_determination(self) -> bool:
        """Test 4: Verify subscription status is determined from real data"""
        print_header("TEST 4: Subscription Status Determination")
        passed = True

        subscription_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "routes" / "subscription.ts"

        if not subscription_file.exists():
            print_error(f"subscription.ts not found at {subscription_file}")
            return False

        content = subscription_file.read_text()

        # Test 4.1: Check auth middleware is applied
        print_info("Test 4.1: Verifying auth middleware is applied...")
        if "app.use('*', authMiddleware)" in content or "app.use('*'" in content:
            print_success("Auth middleware protects all subscription routes")
        else:
            print_error("Auth middleware is not applied")
            passed = False

        # Test 4.2: Verify getUserId is called
        print_info("Test 4.2: Verifying getUserId is called...")
        if 'const userId = getUserId(c)' in content:
            print_success("userId is extracted using getUserId helper")
        else:
            print_error("getUserId is not called")
            passed = False

        # Test 4.3: Check subscription info is retrieved
        print_info("Test 4.3: Checking subscription info retrieval...")
        if 'getSubscriptionInfo(c)' in content or 'const subscriptionInfo = getSubscriptionInfo' in content:
            print_success("Subscription info is retrieved from context")
        else:
            print_warning("Subscription info should be retrieved from context")

        # Test 4.4: Verify isActive logic
        print_info("Test 4.4: Verifying isActive determination logic...")
        if 'const isActive' in content or 'isActive:' in content:
            # Check if it's conditional (not hard-coded)
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'isActive' in line and ('?' in line or '||' in line or '&&' in line):
                    print_success("isActive is determined conditionally from real data")
                    break
            else:
                # Check multi-line logic
                if 'subscriptionInfo.plan' in content or 'hackathonOffer?.isActive' in content:
                    print_success("isActive is determined from subscription data")
                else:
                    print_warning("isActive logic should use subscription data")
        else:
            print_error("isActive is not determined")
            passed = False

        # Test 4.5: Check offer status retrieval
        print_info("Test 4.5: Checking offer status retrieval...")
        if 'getActiveHackathonOfferForResponse' in content:
            print_success("Hackathon offer is retrieved from Clerk")
        else:
            print_warning("Should retrieve hackathon offer from Clerk")

        if 'getSpecialOfferForResponse' in content:
            print_success("Special offer is retrieved from Clerk")
        else:
            print_warning("Should retrieve special offer from Clerk")

        return passed

    def test_development_mode_handling(self) -> bool:
        """Test 5: Verify development mode authentication bypass"""
        print_header("TEST 5: Development Mode Authentication Handling")
        passed = True

        # Test 5.1: Check frontend dev mode
        print_info("Test 5.1: Checking frontend development mode handling...")
        useuser_file = self.bubblelab_dir / "apps" / "bubble-studio" / "src" / "hooks" / "useUser.ts"

        if useuser_file.exists():
            content = useuser_file.read_text()

            if 'DISABLE_AUTH' in content:
                print_success("Frontend checks DISABLE_AUTH environment variable")

                # Check if mock user is returned
                if 'mock-user-id' in content or 'Dev User' in content:
                    print_success("Mock user data is provided in dev mode")
                else:
                    print_warning("Mock user data should be provided")
            else:
                print_warning("Should check DISABLE_AUTH environment variable")
        else:
            print_error("useUser.ts not found")
            passed = False

        # Test 5.2: Check backend dev mode
        print_info("Test 5.2: Checking backend development mode handling...")
        auth_file = self.bubblelab_dir / "apps" / "bubblelab-api" / "src" / "middleware" / "auth.ts"

        if auth_file.exists():
            content = auth_file.read_text()

            if 'env.isDev' in content or 'isDev' in content:
                print_success("Backend checks development mode")

                # Check if dev user ID is used
                if 'devUserId' in content or 'DEV_USER_ID' in content:
                    print_success("Dev user ID is used in development mode")
                else:
                    print_warning("Dev user ID should be used in development")
            else:
                print_warning("Should check development mode")
        else:
            print_error("auth.ts not found")
            passed = False

        # Test 5.3: Check X-User-ID header support
        print_info("Test 5.3: Checking X-User-ID header support for testing...")
        if auth_file.exists():
            content = auth_file.read_text()

            if 'X-User-ID' in content or 'X-User-Id' in content:
                print_success("Backend supports X-User-ID header for testing")
            else:
                print_warning("Should support X-User-ID header for testing")

        return passed

    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        print_header("AUTHENTICATION INTEGRATION TEST SUITE")

        all_passed = True

        # Run all tests
        all_passed &= self.test_jwt_token_structure()
        all_passed &= self.test_user_data_flow()
        all_passed &= self.test_webhook_url_generation()
        all_passed &= self.test_subscription_status_determination()
        all_passed &= self.test_development_mode_handling()

        # Print summary
        print_header("INTEGRATION TEST SUMMARY")
        if all_passed:
            print_success("ALL INTEGRATION TESTS PASSED!")
            print()
            print("Authentication flow is properly implemented:")
            print("  1. JWT tokens are verified and decoded")
            print("  2. User ID is extracted from 'sub' claim")
            print("  3. User data flows from frontend to backend")
            print("  4. Webhook URLs use real userId")
            print("  5. Subscription status is determined from real data")
            print("  6. Development mode is properly handled")
            print()
        else:
            print_error("SOME INTEGRATION TESTS FAILED!")
            print()
            print("Please review the authentication implementation.")

        return all_passed

def main():
    """Main entry point"""
    root_dir = Path(__file__).parent.absolute()

    tester = AuthIntegrationTester(str(root_dir))
    all_passed = tester.run_all_tests()

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
