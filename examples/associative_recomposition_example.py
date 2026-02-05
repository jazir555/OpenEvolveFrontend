"""
Example: Domain-Agnostic Associative Recomposition

Demonstrates the full system with:
1. LLM classifies problem domain (no hardcoded triggers)
2. AgentJSON parses structured output (robust to malformed JSON)
3. Algorithmic verification (content preserved via ground truth)
4. LLM judgment (correctness evaluation)

Usage:
    python associative_recomposition_example.py
"""

import sys
sys.path.insert(0, '.')

from associative_recomposition import AssociativeRecomposer, SolutionType, ProblemDomain


# Mock LLM call function (replace with actual LLM API)
def mock_llm_call(prompt: str) -> str:
    """
    Mock LLM that returns realistic JSON responses.

    In production, replace with actual LLM API call.
    """
    # Simulate LLM analyzing and returning assembly plan
    return """{
    "classification": {
        "problem_type": "User authentication and authorization system",
        "domain": "software_development",
        "solution_type": "code",
        "field": "web security",
        "complexity": "medium",
        "confidence": 0.92,
        "reasoning": "This is a web security problem requiring JWT authentication, user profile management, and role-based access control. The solutions involve Python code for backend implementation."
    },
    "target_solution_type": "code",
    "target_solution_description": "A complete user management system with authentication, profile management, and role-based authorization using JWT tokens.",
    "success_criteria": [
        "All three components (authentication, profile, authorization) are present",
        "JWT implementation is correct with proper token generation",
        "Role-based access control is properly implemented",
        "Code is syntactically correct and executable",
        "Components are properly integrated without duplication"
    ],
    "sub_problem_identities": {
        "sol_1": "JWT authentication module with token generation and validation",
        "sol_2": "User profile management with CRUD operations",
        "sol_3": "Role-based access control middleware"
    },
    "instructions": [
        {
            "sub_problem_id": "sol_1",
            "sub_problem_identity": "JWT authentication module",
            "action": "keep_verbatim",
            "section_header": "Authentication Module",
            "position": 0,
            "preserve_integrity": true,
            "merge_with": null,
            "transformations": null,
            "transition_before": null,
            "transition_after": "With authentication established, we can now manage user profiles.",
            "notes": "Foundation component - must be first"
        },
        {
            "sub_problem_id": "sol_2",
            "sub_problem_identity": "User profile management",
            "action": "keep_verbatim",
            "section_header": "User Profile Management",
            "position": 1,
            "preserve_integrity": true,
            "merge_with": null,
            "transformations": null,
            "transition_before": "Building on the authentication layer,",
            "transition_after": "Finally, we add role-based authorization to secure endpoints.",
            "notes": "Core CRUD functionality for user profiles"
        },
        {
            "sub_problem_id": "sol_3",
            "sub_problem_identity": "Role-based access control",
            "action": "keep_verbatim",
            "section_header": "Role-Based Authorization",
            "position": 2,
            "preserve_integrity": true,
            "merge_with": null,
            "transformations": null,
            "transition_before": "To complete the security layer,",
            "transition_after": null,
            "notes": "Secures endpoints based on user roles"
        }
    ],
    "intro": "This document presents a complete user management system with secure authentication, profile management, and role-based authorization using JWT tokens.",
    "conclusion": "All three components work together to provide a secure, scalable user management system suitable for web applications.",
    "global_notes": "The system uses stateless JWT tokens for scalability. All components preserve code integrity exactly as provided.",
    "confidence_score": 0.92,
    "reasoning": "Authentication must come first as it's the foundation. Profile management follows naturally, using authenticated user IDs. Authorization is added last to secure the endpoints. All components are kept verbatim to preserve code correctness.",
    "estimated_quality": "high"
}"""


def mock_llm_judgment(prompt: str) -> str:
    """
    Mock LLM judgment response.

    In production, this would be an actual LLM evaluating correctness.
    """
    return """{
    "is_correct": true,
    "completeness_score": 0.95,
    "quality_score": 0.90,
    "missing_elements": [],
    "issues": [
        "Could benefit from error handling examples",
        "Consider adding refresh token rotation"
    ],
    "strengths": [
        "All three components present and correctly implemented",
        "Proper JWT usage with signing and validation",
        "Clean separation of concerns",
        "Role-based access control properly integrated"
    ],
    "verdict": "good",
    "confidence": 0.92,
    "reasoning": "The reassembled solution correctly includes all three sub-solutions (authentication, profile management, authorization) in a logical order. The JWT implementation is sound, the profile CRUD operations are complete, and the RBAC middleware is properly integrated. Minor suggestions for enhancement but the core solution is correct and complete."
}"""


def main():
    """Run the associative recomposition example."""
    print("\n" + "="*80)
    print("ASSOCIATIVE RECOMPOSITION EXAMPLE")
    print("Domain-Agnostic System with LLM Judge + AgentJSON + Verification")
    print("="*80 + "\n")

    # Create sub-solutions (these would come from problem decomposition)
    sub_solutions = {
        'sol_1': {
            'description': 'JWT Authentication',
            'dependencies': [],
            'confidence_score': 0.95,
            'solution_content': '''```python
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional

def generate_token(user_id: int, secret: str) -> str:
    """Generate JWT token for user."""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, secret, algorithm='HS256')

def verify_token(token: str, secret: str) -> Optional[Dict]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
```'''
        },
        'sol_2': {
            'description': 'User Profile Management',
            'dependencies': ['sol_1'],
            'confidence_score': 0.90,
            'solution_content': '''```python
from typing import Dict, Any, Optional
from datetime import datetime

class UserProfile:
    """User profile data model."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.email = None
        self.full_name = None
        self.preferences = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def update_profile(self, data: Dict[str, Any]) -> None:
        """Update user profile fields."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'full_name': self.full_name,
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
```'''
        },
        'sol_3': {
            'description': 'Role-Based Access Control',
            'dependencies': ['sol_1'],
            'confidence_score': 0.88,
            'solution_content': '''```python
from functools import wraps
from typing import Callable, List

def require_roles(allowed_roles: List[str]):
    """Decorator to require specific roles for endpoint access."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from request
            token = get_token_from_request()
            if not token:
                return {'error': 'No token provided'}, 401

            # Verify token and get user roles
            payload = verify_token(token, SECRET_KEY)
            if not payload:
                return {'error': 'Invalid token'}, 401

            user_roles = get_user_roles(payload['user_id'])
            if not any(role in allowed_roles for role in user_roles):
                return {'error': 'Insufficient permissions'}, 403

            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage:
# @require_roles(['admin', 'moderator'])
# def delete_user(user_id: int):
#     # Only admins and moderators can delete users
#     pass
```'''
        }
    }

    # Mock conflicts
    conflicts = [
        {
            'conflict_type': 'dependency',
            'description': 'sol_2 and sol_3 both depend on sol_1 (authentication)',
            'severity': 'low'
        }
    ]

    # Problem statement
    problem_statement = """
    Build a complete user management system for a web application with:
    1. Secure JWT-based authentication
    2. User profile management with CRUD operations
    3. Role-based access control to protect endpoints
    """

    print(f"Problem: {problem_statement.strip()}\n")
    print(f"Sub-solutions: {len(sub_solutions)}")
    print(f"Conflicts: {len(conflicts)}\n")

    # Create recomposer
    recomposer = AssociativeRecomposer(
        use_agentjson=True,  # Use AgentJSON for robust JSON parsing
        max_retries=3
    )

    # Mock LLM call that uses our mock responses
    call_count = [0]
    def llm_call_fn(prompt: str) -> str:
        call_count[0] += 1
        if "JUDGE" in prompt:
            print(f"-> LLM Call #{call_count[0]}: Judgment")
            return mock_llm_judgment(prompt)
        else:
            print(f"-> LLM Call #{call_count[0]}: Assembly Plan")
            return mock_llm_call(prompt)

    # Run recomposition
    print("-"*80)
    print("STARTING RECOMPOSITION PIPELINE")
    print("-"*80 + "\n")

    assembled, metadata = recomposer.recompose_with_verification(
        sub_solutions=sub_solutions,
        conflicts=conflicts,
        problem_statement=problem_statement,
        llm_call_fn=llm_call_fn
    )

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    if assembled:
        print("[OK] RECOMPOSITION SUCCESSFUL\n")

        # Display classification
        if metadata.get('classification'):
            print("CLASSIFICATION:")
            cls = metadata['classification']
            print(f"  Domain:       {cls['domain']}")
            print(f"  Type:         {cls['solution_type']}")
            print(f"  Field:        {cls['field']}")
            print(f"  Complexity:   {cls['complexity']}")
            print(f"  Confidence:   {cls['confidence']:.2f}")
            print(f"  Reasoning:    {cls['reasoning'][:80]}...")
            print()

        # Display judgment
        if metadata.get('judgment'):
            print("LLM JUDGMENT:")
            j = metadata['judgment']
            print(f"  Is Correct:       {j.get('is_correct', False)}")
            print(f"  Completeness:     {j.get('completeness_score', 0):.2f}")
            print(f"  Quality:          {j.get('quality_score', 0):.2f}")
            print(f"  Verdict:          {j.get('verdict', 'unknown')}")
            print(f"  Confidence:       {j.get('confidence', 0):.2f}")
            print()

            if j.get('strengths'):
                print("  Strengths:")
                for strength in j['strengths']:
                    print(f"    - {strength}")
                print()

            if j.get('issues'):
                print("  Issues:")
                for issue in j['issues']:
                    print(f"    - {issue}")
                print()

        # Display verification results
        if metadata.get('verification_results'):
            print("ALGORITHMIC VERIFICATION:")
            for sub_id, (preserved, details) in metadata['verification_results'].items():
                status = "[OK]" if preserved else "[FAIL]"
                print(f"  {status} {sub_id}: {details}")
            print()

        # Display assembled solution
        print("-"*80)
        print("ASSEMBLED SOLUTION (first 800 chars)")
        print("-"*80)
        print(assembled[:800] + "..." if len(assembled) > 800 else assembled)

    else:
        print("[FAIL] RECOMPOSITION FAILED\n")
        print(f"Metadata: {metadata}")

    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
