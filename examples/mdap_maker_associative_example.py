"""
MDAP/MAKER + Associative Recomposition Example

Demonstrates the complete integrated system with:
- MAKER workflow orchestration
- Associative recomposition (domain-agnostic LLM)
- MDAP multi-agent validation
- Ground truth verification

Usage:
    python mdap_maker_associative_example.py
"""

import sys
sys.path.insert(0, '.')

from mdap_maker_associative_integration import (
    MakerRecomposerWorkflow,
    recompose_with_mdap_maker
)


# Mock LLM functions
def mock_primary_llm(prompt: str) -> str:
    """Mock primary LLM for assembly planning"""
    if "JUDGE" in prompt:
        # Judgment response
        return """{
    "is_correct": true,
    "completeness_score": 0.95,
    "quality_score": 0.90,
    "missing_elements": [],
    "issues": ["Could add more error handling"],
    "strengths": ["All components present and correct"],
    "verdict": "good",
    "confidence": 0.92,
    "reasoning": "The solution correctly includes all three sub-solutions (authentication, profile management, authorization) with proper JWT implementation and role-based access control."
}"""
    else:
        # Assembly plan response
        return """{
    "classification": {
        "problem_type": "User authentication and authorization system",
        "domain": "software_development",
        "solution_type": "code",
        "field": "web security",
        "complexity": "medium",
        "confidence": 0.92,
        "reasoning": "This is a web security problem requiring JWT authentication, user profile management, and role-based access control using Python."
    },
    "target_solution_type": "code",
    "target_solution_description": "A complete user management system with JWT-based authentication, user profile CRUD operations, and role-based access control middleware.",
    "success_criteria": [
        "All three components (authentication, profile, authorization) are present",
        "JWT implementation is correct with proper signing and validation",
        "Role-based access control properly secures endpoints",
        "Code is syntactically correct and executable",
        "Components are integrated without duplication or conflicts"
    ],
    "sub_problem_identities": {
        "sol_1": "JWT authentication module with token generation and validation",
        "sol_2": "User profile data model with CRUD operations",
        "sol_3": "Role-based access control decorator for endpoint protection"
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
            "notes": "Foundation component - provides JWT tokens for authentication"
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
            "transition_after": "Finally, we add role-based authorization to secure sensitive endpoints.",
            "notes": "Core CRUD functionality for managing user profiles"
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
            "notes": "Protects endpoints based on user roles using JWT claims"
        }
    ],
    "intro": "This document presents a complete user management system with secure JWT-based authentication, user profile management, and role-based access control suitable for web applications.",
    "conclusion": "All three components work together to provide a secure, scalable user management system. The JWT tokens are stateless for scalability, the profile model is flexible for customization, and the RBAC middleware provides flexible authorization.",
    "global_notes": "The system uses Python with JWT for stateless authentication. All components preserve code integrity exactly as provided by the LLM generation.",
    "confidence_score": 0.92,
    "reasoning": "Authentication must come first as it's the foundation. Profile management follows naturally, using authenticated user IDs from JWT tokens. Authorization is added last to protect the secured endpoints. All components are kept verbatim to preserve code correctness.",
    "estimated_quality": "high"
}"""


def mock_mdap_agent_1(prompt: str) -> str:
    """Mock MDAP Agent 1 - Security focused"""
    return """{
    "vote": "approve",
    "confidence": 0.90,
    "completeness_score": 0.95,
    "quality_score": 0.88,
    "correctness_score": 0.92,
    "missing_elements": [],
    "issues_found": ["JWT secret key should be from environment variable", "Consider adding refresh token rotation"],
    "strengths_found": ["Proper JWT implementation with HS256", "Correct role extraction from token", "Clean separation of concerns"],
    "red_flags": [],
    "reasoning": "The authentication module correctly implements JWT with proper signing. The RBAC decorator correctly checks roles. The profile model is well-structured. Minor security improvements suggested but overall correct and complete."
}"""


def mock_mdap_agent_2(prompt: str) -> str:
    """Mock MDAP Agent 2 - Architecture focused"""
    return """{
    "vote": "approve",
    "confidence": 0.88,
    "completeness_score": 0.92,
    "quality_score": 0.90,
    "correctness_score": 0.89,
    "missing_elements": [],
    "issues_found": ["Consider adding database integration examples", "Add input validation for profile updates"],
    "strengths_found": ["Good modular design", "Clear component boundaries", "Proper dependency ordering"],
    "red_flags": [],
    "reasoning": "The solution follows good architectural principles with clear separation between authentication, profile management, and authorization. The ordering of components makes sense (auth -> profile -> authz). The code is production-ready with minor enhancements needed."
}"""


def mock_mdap_agent_3(prompt: str) -> str:
    """Mock MDAP Agent 3 - Code quality focused"""
    return """{
    "vote": "approve",
    "confidence": 0.95,
    "completeness_score": 0.95,
    "quality_score": 0.92,
    "correctness_score": 0.95,
    "missing_elements": [],
    "issues_found": ["Add docstring examples", "Consider adding type hints for better IDE support"],
    "strengths_found": ["Clean Python code", "Proper error handling", "Good naming conventions"],
    "red_flags": [],
    "reasoning": "All three components are production-ready Python code with proper error handling and clean syntax. The JWT implementation follows best practices. The profile data model is well-designed. The RBAC decorator is properly implemented with functools.wraps."
}"""


def mock_mdap_agent_4(prompt: str) -> str:
    """Mock MDAP Agent 4 - Integration focused"""
    return """{
    "vote": "approve",
    "confidence": 0.85,
    "completeness_score": 0.90,
    "quality_score": 0.87,
    "correctness_score": 0.88,
    "missing_elements": [],
    "issues_found": ["Transitions could be more detailed", "Consider adding API usage examples"],
    "strengths_found": ["Smooth integration between components", "Clear transition text", "Logical flow of components"],
    "red_flags": [],
    "reasoning": "The components integrate well with clear transitions. The authentication module correctly feeds into profile management, and authorization properly secures the system. The assembly maintains the integrity of all original code while creating a coherent final solution."
}"""


def mock_mdap_agent_5(prompt: str) -> str:
    """Mock MDAP Agent 5 - Requirements focused"""
    return """{
    "vote": "approve",
    "confidence": 0.92,
    "completeness_score": 0.95,
    "quality_score": 0.90,
    "correctness_score": 0.93,
    "missing_elements": [],
    "issues_found": ["Consider adding rate limiting", "Add password strength validation"],
    "strengths_found": ["All requirements met", "Complete feature set", "Proper security implementation"],
    "red_flags": [],
    "reasoning": "The solution meets all stated requirements: authentication with JWT, user profile management with CRUD, and role-based access control. All success criteria are satisfied. The solution is correct, complete, and production-ready."
}"""


def main():
    """Run the MDAP/MAKER + Associative Recomposition example."""
    print("\n" + "="*80)
    print("MDAP/MAKER + ASSOCIATIVE RECOMPOSITION EXAMPLE")
    print("="*80 + "\n")

    # Sub-solutions (from decomposition)
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
```'''
        }
    }

    # Conflicts
    conflicts = [
        {
            'conflict_type': 'dependency',
            'description': 'sol_2 and sol_3 both depend on sol_1',
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

    print(f"Problem: {problem_statement.strip()}")
    print(f"Sub-solutions: {len(sub_solutions)}")
    print(f"Conflicts: {len(conflicts)}\n")

    # Run full workflow
    print("-"*80)
    print("RUNNING FULL MDAP/MAKER WORKFLOW")
    print("-"*80 + "\n")

    workflow = MakerRecomposerWorkflow(
        use_mdap=True,
        use_associative=True,
        num_mdap_agents=5
    )

    results = workflow.run_full_workflow(
        problem_statement=problem_statement,
        sub_solutions=sub_solutions,
        conflicts=conflicts,
        llm_call_fn=mock_primary_llm,
        mdap_agent_llm_calls=[
            mock_mdap_agent_1,
            mock_mdap_agent_2,
            mock_mdap_agent_3,
            mock_mdap_agent_4,
            mock_mdap_agent_5
        ]
    )

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print(f"Success: {results['success']}")
    print(f"Stages Completed: {', '.join(results['workflow_stages'])}\n")

    # Initial assessment
    if 'initial_assessment' in results['metadata']:
        print("INITIAL ASSESSMENT:")
        ia = results['metadata']['initial_assessment']
        print(f"  Sub-solutions: {ia['num_sub_solutions']}")
        print(f"  Conflicts: {ia['num_conflicts']}")
        print(f"  Contains code: {ia['has_code']}")
        print(f"  Complexity: {ia['estimated_complexity']}\n")

    # Associative recomposition
    if 'associative_recomposition' in results['metadata']:
        print("ASSOCIATIVE RECOMPOSITION:")
        ar = results['metadata']['associative_recomposition']

        if 'classification' in ar:
            cls = ar['classification']
            print(f"  Domain: {cls['domain']}")
            print(f"  Type: {cls['solution_type']}")
            print(f"  Field: {cls['field']}")
            print(f"  Complexity: {cls['complexity']}\n")

        if 'judgment' in ar and ar['judgment'] is not None:
            print("  LLM JUDGMENT:")
            j = ar['judgment']
            print(f"    Is Correct: {j.get('is_correct', 'N/A')}")
            print(f"    Quality: {j.get('quality_score', 0):.2f}")
            print(f"    Verdict: {j.get('verdict', 'N/A')}\n")
        elif 'judgment' in ar:
            print("  LLM JUDGMENT:")
            print("    Status: Failed to parse judgment\n")

    # Algorithmic verification
    if 'algorithmic_verification' in results['metadata']:
        print("ALGORITHMIC VERIFICATION:")
        av = results['metadata']['algorithmic_verification']
        print(f"  All Preserved: {av['all_preserved']}")

        if 'verification_results' in av:
            for sub_id, (preserved, details) in av['verification_results'].items():
                status = "[OK]" if preserved else "[X]"
                print(f"    {status} {sub_id}: {details}\n")

    # MDAP validation
    if 'mdap_validation' in results['metadata']:
        print("MDAP MULTI-AGENT VALIDATION:")
        mdap = results['metadata']['mdap_validation']

        print(f"  Agents: {mdap['num_agents']}")
        print(f"  Consensus: {mdap['consensus']['decision']}")
        print(f"  Votes For: {mdap['consensus']['votes_for']}")
        print(f"  Votes Against: {mdap['consensus']['votes_against']}")
        print(f"  Agreement: {mdap['agreement_ratio']:.2%}\n")

        if 'validation_details' in mdap:
            vd = mdap['validation_details']
            print("  Validation Metrics:")
            print(f"    Avg Confidence: {vd['avg_confidence']:.2f}")
            print(f"    Avg Quality: {vd['avg_quality']:.2f}")
            print(f"    Avg Correctness: {vd['avg_correctness']:.2f}\n")

    # Final assembled solution
    if results['final_assembled']:
        print("-"*80)
        print("FINAL ASSEMBLED SOLUTION")
        print("-"*80)
        print(results['final_assembled'][:1500] + "..." if len(results['final_assembled']) > 1500 else results['final_assembled'])

    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
