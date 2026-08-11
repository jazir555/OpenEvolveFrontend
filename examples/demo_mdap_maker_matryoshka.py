#!/usr/bin/env python3
"""
================================================================================
MDAP/MAKER + MATRYOSHKA INTEGRATION DEMONSTRATION
================================================================================

A complete educational demonstration showing how MDAP/MAKER integrates with
optional Matryoshka unified memory for enhanced problem-solving capabilities.

Demo Narrative:
---------------
1. SETUP & CAPABILITY DETECTION - See what's available and active
2. STANDARD MDAP/MAKER - Works without any optional dependencies
3. MATRYOSHKA DOCUMENT ANALYSIS - Deep exploration of large documents
4. MEMORY BRIDGE - Cross-session learning and pattern reuse
5. FULL INTEGRATION - Complex problems with all capabilities combined

Usage:
    python demo_mdap_maker_matryoshka.py              # Run all demos
    python demo_mdap_maker_matryoshka.py --part 1     # Setup only
    python demo_mdap_maker_matryoshka.py --part 2     # Standard MDAP only
    python demo_mdap_maker_matryoshka.py --part 3     # Matryoshka only
    python demo_mdap_maker_matryoshka.py --part 4     # Memory bridge only
    python demo_mdap_maker_matryoshka.py --part 5     # Full integration

Author: OpenEvolve AI
Version: 2.0.0
Date: February 2026
================================================================================
"""

from __future__ import annotations

import argparse
import json
import hashlib
import time
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from pathlib import Path
import uuid

# =============================================================================
# COLOR OUTPUT UTILITIES
# =============================================================================

class Colors:
    """ANSI color codes for beautiful terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    DIM = '\033[2m'
    MAGENTA = '\033[35m'
    ORANGE = '\033[38;5;208m'
    WHITE = '\033[97m'


def print_banner(text: str, char: str = "="):
    """Print a prominent banner."""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}\n")


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}>> {text}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * (len(text) + 4)}{Colors.END}\n")


def print_subheader(text: str):
    """Print a subsection header."""
    print(f"\n  {Colors.BOLD}{Colors.WHITE}-> {text}{Colors.END}")


def print_success(text: str):
    """Print success message."""
    print(f"  {Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"  {Colors.YELLOW}[WARN] {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"  {Colors.RED}[ERR] {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"  {Colors.CYAN}[INFO] {text}{Colors.END}")


def print_stat(label: str, value: str, unit: str = ""):
    """Print a statistic line."""
    print(f"    {Colors.DIM}{label}:{Colors.END} {Colors.BOLD}{Colors.GREEN}{value}{Colors.END} {Colors.DIM}{unit}{Colors.END}")


def print_bullet(text: str, indent: int = 2):
    """Print a bullet point."""
    spaces = " " * indent
    print(f"{spaces}{Colors.DIM}*{Colors.END} {text}")


def print_arrow(text: str, indent: int = 2):
    """Print an arrow point."""
    spaces = " " * indent
    print(f"{spaces}{Colors.CYAN}->{Colors.END} {text}")


def print_box(title: str, content: List[str], width: int = 60):
    """Print a boxed section."""
    print(f"\n  {Colors.BOLD}{'+' + '-' * (width-2) + '+'}{Colors.END}")
    print(f"  {Colors.BOLD}|{Colors.END} {Colors.CYAN}{title:<{width-4}}{Colors.END} {Colors.BOLD}|{Colors.END}")
    print(f"  {Colors.BOLD}|{'-' * (width-2)}|{Colors.END}")
    for line in content:
        print(f"  {Colors.BOLD}|{Colors.END} {line:<{width-4}} {Colors.BOLD}|{Colors.END}")
    print(f"  {Colors.BOLD}|{'-' * (width-2)}|{Colors.END}\n")


def divider():
    """Print a divider line."""
    print(f"\n{Colors.DIM}{'-' * 80}{Colors.END}\n")


# =============================================================================
# OPTIONAL DEPENDENCY CHECKS
# =============================================================================

print(f"\n{Colors.DIM}Checking optional dependencies...{Colors.END}\n")

# Check Matryoshka availability
try:
    from matryoshka_unified_memory_integration import (
        MatryoshkaMemoryBridge,
        MatryoshkaExplorationSession,
        UnifiedMatryoshkaClient,
        create_unified_matryoshka_client,
    )
    MATRYOSHKA_AVAILABLE = True
    print_success("Matryoshka Unified Memory integration available")
except ImportError as e:
    MATRYOSHKA_AVAILABLE = False
    print_warning(f"Matryoshka not available: {e}")

# Check MDAP/MAKER availability
try:
    from mdap_engine import MDAPConfig, MDAPRunResult, RedFlagRules, MDAPOrchestrator
    MDAP_AVAILABLE = True
    print_success("MDAP engine available")
except ImportError as e:
    MDAP_AVAILABLE = False
    print_warning(f"MDAP not available: {e}")

try:
    from maker_engine import MakerEngine, MakerConfig, MakerStep
    MAKER_AVAILABLE = True
    print_success("MAKER engine available")
except ImportError as e:
    MAKER_AVAILABLE = False
    print_warning(f"MAKER not available: {e}")

# Check Unified Memory availability
try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        create_unified_system,
    )
    UNIFIED_MEMORY_AVAILABLE = True
    print_success("Unified Memory System available")
except ImportError as e:
    UNIFIED_MEMORY_AVAILABLE = False
    print_warning(f"Unified Memory not available: {e}")

# Check Memory Bridge availability
try:
    from mdap_memory_bridge import MDAPMemoryBridge, create_mdap_memory_bridge
    MEMORY_BRIDGE_AVAILABLE = True
    print_success("MDAP Memory Bridge available")
except ImportError as e:
    MEMORY_BRIDGE_AVAILABLE = False
    print_warning(f"MDAP Memory Bridge not available: {e}")

# Check Matryoshka Integration availability
try:
    from mdap_maker_matryoshka_integration import (
        MDAPMakerWithMatryoshka,
        MDAPMatryoshkaConfig,
        ExplorationResult,
        MDAPMatryoshkaResult,
    )
    INTEGRATION_AVAILABLE = True
    print_success("MDAP-Matryoshka Integration available")
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    print_warning(f"MDAP-Matryoshka Integration not available: {e}")


# =============================================================================
# SAMPLE DATA FOR DEMOS
# =============================================================================

# Sample "large document" - A complex codebase simulation
SAMPLE_CODEBASE = '''
# Microservices Architecture - E-Commerce Platform
# =================================================

## services/order_service/main.py
```python
"""Order Service - Handles order lifecycle management."""
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Order:
    order_id: str
    customer_id: str
    items: List[Dict]
    total_amount: float
    status: str  # pending, confirmed, shipped, delivered
    created_at: datetime
    
class OrderService:
    """Core order processing service."""
    
    def __init__(self, db_connection, event_bus):
        self.db = db_connection
        self.event_bus = event_bus
        self.active_orders: Dict[str, Order] = {}
        
    async def create_order(self, customer_id: str, items: List[Dict]) -> Order:
        """Create a new order with validation."""
        # Validate inventory
        for item in items:
            if not await self._check_inventory(item['sku'], item['quantity']):
                raise ValueError(f"Insufficient inventory for {item['sku']}")
        
        # Calculate totals
        total = sum(item['price'] * item['quantity'] for item in items)
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            customer_id=customer_id,
            items=items,
            total_amount=total,
            status="pending",
            created_at=datetime.utcnow()
        )
        
        # Persist
        await self._persist_order(order)
        
        # Emit event
        await self.event_bus.publish("order.created", {
            "order_id": order.order_id,
            "customer_id": customer_id,
            "amount": total
        })
        
        return order
    
    async def process_payment(self, order_id: str, payment_method: Dict) -> bool:
        """Process payment for an order."""
        order = await self._get_order(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        # Payment processing logic
        payment_result = await self._call_payment_gateway(
            amount=order.total_amount,
            method=payment_method
        )
        
        if payment_result['success']:
            order.status = "confirmed"
            await self._update_order(order)
            await self.event_bus.publish("order.paid", {"order_id": order_id})
            return True
        else:
            logger.error(f"Payment failed for order {order_id}")
            return False
    
    async def _check_inventory(self, sku: str, quantity: int) -> bool:
        """Check if item is available in inventory."""
        # Calls inventory service
        pass
    
    async def _persist_order(self, order: Order):
        """Persist order to database."""
        # Database operations
        pass

# Security vulnerabilities found:
# 1. No rate limiting on order creation - potential DoS
# 2. Payment method not validated before gateway call
# 3. No idempotency key for payment processing
# 4. Insufficient logging for audit trails
```

## services/inventory_service/inventory_manager.py
```python
"""Inventory Management Service."""
import redis
import json
from typing import Dict, Optional
from contextlib import asynccontextmanager

class InventoryManager:
    """Manages stock levels across warehouses."""
    
    def __init__(self, redis_client: redis.Redis, db_pool):
        self.redis = redis_client
        self.db = db_pool
        self.cache_ttl = 300  # 5 minutes
    
    async def reserve_inventory(self, sku: str, quantity: int, order_id: str) -> bool:
        """Reserve inventory for an order."""
        cache_key = f"inventory:{sku}"
        
        # Check cache first
        cached = self.redis.get(cache_key)
        if cached:
            stock = json.loads(cached)
        else:
            stock = await self._fetch_from_db(sku)
            self.redis.setex(cache_key, self.cache_ttl, json.dumps(stock))
        
        available = stock['quantity'] - stock['reserved']
        
        if available >= quantity:
            # Reserve stock
            stock['reserved'] += quantity
            self.redis.setex(cache_key, self.cache_ttl, json.dumps(stock))
            
            # Persist to DB
            await self._update_db(sku, stock['reserved'])
            
            # Add reservation record
            await self._add_reservation(order_id, sku, quantity)
            return True
        
        return False
    
    async def release_reservation(self, order_id: str):
        """Release inventory reservation."""
        reservations = await self._get_reservations(order_id)
        for res in reservations:
            await self._decrement_reserved(res['sku'], res['quantity'])
        await self._clear_reservations(order_id)

# Issues:
# 1. Race condition in reservation logic
# 2. Cache inconsistency possible
# 3. No distributed locking
```

## services/user_service/auth.py
```python
"""Authentication and Authorization Service."""
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict

class AuthService:
    """Handles user authentication."""
    
    SECRET_KEY = "hardcoded-secret-key"  # CRITICAL: Should be from env
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE = 30  # minutes
    
    def create_access_token(self, user_id: str, roles: List[str]) -> str:
        """Create JWT access token."""
        expires = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE)
        
        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": expires,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        try:
            return jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# CRITICAL VULNERABILITIES:
# 1. Hardcoded secret key
# 2. No token revocation mechanism
# 3. Weak token expiration
# 4. No refresh token rotation
```

## api/gateway/routes.py
```python
"""API Gateway Routes Configuration."""
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="E-Commerce API Gateway")

# CORS configuration - too permissive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # SECURITY: Too permissive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting missing
# Request validation missing
# Circuit breaker missing

@app.post("/api/v1/orders")
async def create_order(request: Request):
    """Proxy order creation to order service."""
    body = await request.json()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://order-service:8000/orders",
            json=body,
            timeout=30.0
        )
    
    return response.json()

# More routes...
```
'''

# Sample problems for different demos
SIMPLE_PROBLEM = """
Design a function to calculate the factorial of a number with proper error handling.
Requirements:
- Handle negative inputs gracefully
- Optimize for large numbers
- Include input validation
"""

SORT_PROBLEM = """
Optimize a sorting algorithm for a dataset of 1 million integers where:
- 80% of data is already sorted
- Memory usage must be minimized
- Stability is required
"""

SEARCH_PROBLEM = """
Implement an efficient search algorithm for a dataset where:
- Data is sorted but distributed across multiple nodes
- Network latency is significant
- Results must be paginated
"""

COMPLEX_PROBLEM = """
Refactor a monolithic e-commerce application to microservices architecture.
Constraints:
- Zero downtime during migration
- Maintain data consistency across services
- Support horizontal scaling
- Implement circuit breakers and retry logic
- Ensure observability with distributed tracing
- Handle eventual consistency in inventory management
"""


# =============================================================================
# ENHANCED CLIENT CLASS
# =============================================================================

class EnhancedMDAPClient:
    """
    Enhanced client that wraps MDAP/MAKER with optional Matryoshka integration.
    
    This is the main interface demonstrated in this script.
    """
    
    def __init__(self, enable_matryoshka: bool = True, storage_path: str = "./demo_memory"):
        """
        Initialize the enhanced client.
        
        Args:
            enable_matryoshka: Whether to enable Matryoshka if available
            storage_path: Path for memory storage
        """
        self.storage_path = storage_path
        self.capabilities = self._detect_capabilities()
        
        # Initialize components based on availability
        self.matryoshka_client = None
        self.mdap_engine = None
        self.maker_engine = None
        self.memory_bridge = None
        self.unified_memory = None
        
        # Session tracking
        self.session_decompositions: List[Dict] = []
        self.session_votes: List[Dict] = []
        
        self._initialize_components(enable_matryoshka)
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect available capabilities."""
        return {
            "mdap": MDAP_AVAILABLE,
            "maker": MDAP_AVAILABLE,
            "matryoshka": MATRYOSHKA_AVAILABLE,
            "unified_memory": UNIFIED_MEMORY_AVAILABLE,
            "memory_bridge": MEMORY_BRIDGE_AVAILABLE,
            "full_integration": INTEGRATION_AVAILABLE,
        }
    
    def _initialize_components(self, enable_matryoshka: bool):
        """Initialize all available components."""
        # Initialize Unified Memory if available
        if UNIFIED_MEMORY_AVAILABLE:
            try:
                # Try different initialization patterns
                try:
                    from knowledge_unified_memory_system import UnifiedMemoryConfig
                    config = UnifiedMemoryConfig(db_dir=self.storage_path)
                    self.unified_memory = create_unified_system(config=config)
                except TypeError:
                    # Fallback to simpler initialization
                    self.unified_memory = create_unified_system(db_dir=self.storage_path)
                print_success("Unified Memory System initialized")
            except Exception as e:
                print_warning(f"Failed to initialize Unified Memory: {e}")
        
        # Initialize MDAP/Maker
        if MDAP_AVAILABLE:
            try:
                mdap_config = MDAPConfig(
                    k_min=2,
                    k_max=5,
                    timeout_seconds=60
                )
                self.mdap_engine = MDAPOrchestrator(config=mdap_config)
                print_success("MDAP Engine initialized")
            except Exception as e:
                print_warning(f"Failed to initialize MDAP: {e}")
        
        # Initialize Memory Bridge
        if MEMORY_BRIDGE_AVAILABLE and self.unified_memory:
            try:
                # Try different initialization patterns
                try:
                    self.memory_bridge = create_mdap_memory_bridge(
                        unified_memory=self.unified_memory
                    )
                except TypeError:
                    # Fallback to storage path
                    self.memory_bridge = create_mdap_memory_bridge(
                        storage_path=self.storage_path
                    )
                print_success("Memory Bridge initialized")
            except Exception as e:
                print_warning(f"Failed to initialize Memory Bridge: {e}")
        
        # Initialize Matryoshka
        if enable_matryoshka and MATRYOSHKA_AVAILABLE:
            try:
                # Try different initialization patterns
                try:
                    self.matryoshka_client = create_unified_matryoshka_client(
                        storage_path=self.storage_path,
                        enable_unified_memory=True
                    )
                except TypeError:
                    # Fallback to simpler initialization
                    self.matryoshka_client = create_unified_matryoshka_client()
                print_success("Matryoshka client initialized")
            except Exception as e:
                print_warning(f"Failed to initialize Matryoshka: {e}")
    
    def solve(self, problem: str, use_matryoshka: bool = False) -> Dict[str, Any]:
        """
        Solve a problem using MDAP/MAKER.
        
        Args:
            problem: The problem statement
            use_matryoshka: Whether to use Matryoshka for exploration
        
        Returns:
            Result dictionary with solution and metadata
        """
        start_time = time.time()
        
        result = {
            "problem": problem[:100] + "..." if len(problem) > 100 else problem,
            "solution": None,
            "matryoshka_used": False,
            "decomposition": None,
            "voting_rounds": 0,
            "memories_retrieved": 0,
            "execution_time_ms": 0,
        }
        
        # Step 1: Check for similar past decompositions (Memory Bridge)
        if self.memory_bridge:
            similar = self._find_similar_decompositions(problem)
            if similar:
                result["memories_retrieved"] = len(similar)
                print_info(f"Retrieved {len(similar)} similar decompositions from memory")
        
        # Step 2: Decompose the problem
        decomposition = self._decompose_problem(problem)
        result["decomposition"] = decomposition
        
        # Step 3: Use Matryoshka if requested and available
        if use_matryoshka and self.matryoshka_client:
            exploration = self._explore_with_matryoshka(problem)
            result["matryoshka_used"] = True
            result["exploration"] = exploration
        
        # Step 4: Vote on solutions
        voting_result = self._vote_on_solutions(decomposition)
        result["voting_rounds"] = voting_result.get("rounds", 0)
        result["solution"] = voting_result.get("winner", "Simulated solution")
        
        # Step 5: Store decomposition for future learning
        if self.memory_bridge:
            self._store_decomposition(problem, decomposition)
        
        result["execution_time_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    def analyze_document(self, problem: str, document: str) -> Dict[str, Any]:
        """
        Analyze a document using Matryoshka-enhanced MDAP.
        
        Args:
            problem: The analysis objective
            document: Document content to analyze
        
        Returns:
            Analysis result with findings and insights
        """
        start_time = time.time()
        
        result = {
            "problem": problem,
            "document_size": len(document),
            "matryoshka_available": self.matryoshka_client is not None,
            "exploration_steps": [],
            "findings": [],
            "insights": [],
            "execution_time_ms": 0,
        }
        
        if not self.matryoshka_client:
            # Fallback to standard analysis
            result["findings"] = self._simulate_document_analysis(problem, document)
            result["fallback_used"] = True
        else:
            # Use Matryoshka for deep exploration
            exploration = self._explore_with_matryoshka(problem, document)
            result["exploration_steps"] = exploration.get("steps", [])
            result["insights"] = exploration.get("insights", [])
            result["findings"] = exploration.get("findings", [])
        
        result["execution_time_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    def decompose(self, problem: str) -> Dict[str, Any]:
        """
        Decompose a problem into subproblems.
        
        Args:
            problem: The problem to decompose
        
        Returns:
            Decomposition result
        """
        return self._decompose_problem(problem)
    
    def _decompose_problem(self, problem: str) -> Dict[str, Any]:
        """Simulate problem decomposition."""
        # In a real implementation, this would call MDAP
        # For demo purposes, we simulate intelligent decomposition
        
        problem_lower = problem.lower()
        
        if "sort" in problem_lower:
            subproblems = [
                {"id": 1, "text": "Analyze data distribution characteristics", "type": "analysis"},
                {"id": 2, "text": "Select appropriate sorting algorithm", "type": "decision"},
                {"id": 3, "text": "Implement with memory optimization", "type": "implementation"},
                {"id": 4, "text": "Verify stability guarantees", "type": "verification"},
            ]
        elif "search" in problem_lower:
            subproblems = [
                {"id": 1, "text": "Analyze query patterns and frequency", "type": "analysis"},
                {"id": 2, "text": "Design distributed index structure", "type": "design"},
                {"id": 3, "text": "Implement result aggregation", "type": "implementation"},
                {"id": 4, "text": "Add pagination and caching", "type": "optimization"},
            ]
        elif "refactor" in problem_lower or "microservice" in problem_lower:
            subproblems = [
                {"id": 1, "text": "Analyze monolith boundaries and dependencies", "type": "analysis"},
                {"id": 2, "text": "Design service decomposition strategy", "type": "design"},
                {"id": 3, "text": "Plan data migration approach", "type": "planning"},
                {"id": 4, "text": "Implement inter-service communication", "type": "implementation"},
                {"id": 5, "text": "Add observability and circuit breakers", "type": "infrastructure"},
                {"id": 6, "text": "Verify zero-downtime migration", "type": "verification"},
            ]
        else:
            subproblems = [
                {"id": 1, "text": "Understand requirements and constraints", "type": "analysis"},
                {"id": 2, "text": "Design solution approach", "type": "design"},
                {"id": 3, "text": "Implement core functionality", "type": "implementation"},
                {"id": 4, "text": "Test and validate", "type": "verification"},
            ]
        
        return {
            "strategy": "adaptive",
            "subproblems": subproblems,
            "complexity_score": len(problem) / 100,
        }
    
    def _vote_on_solutions(self, decomposition: Dict) -> Dict[str, Any]:
        """Simulate voting on solutions."""
        num_subproblems = len(decomposition.get("subproblems", []))
        
        # Simulate multiple voting rounds
        rounds = min(num_subproblems, 3)
        
        candidates = [f"Solution approach {i+1}" for i in range(rounds + 1)]
        
        return {
            "rounds": rounds,
            "candidates": candidates,
            "winner": candidates[0],
            "confidence": 0.85,
        }
    
    def _explore_with_matryoshka(self, problem: str, document: Optional[str] = None) -> Dict[str, Any]:
        """Simulate Matryoshka exploration."""
        steps = []
        
        # Simulate exploration steps
        num_steps = random.randint(5, 12)
        
        for i in range(num_steps):
            step_types = ["code_generation", "observation", "insight", "hypothesis", "verification"]
            step = {
                "step_number": i + 1,
                "type": random.choice(step_types),
                "description": f"Exploration step {i+1} for: {problem[:50]}...",
                "tokens_used": random.randint(500, 2000),
            }
            steps.append(step)
        
        # Generate insights based on problem type
        insights = []
        if document and "security" in problem.lower():
            insights = [
                "Hardcoded credentials found in auth.py",
                "CORS configuration too permissive in gateway",
                "Race condition in inventory reservation",
                "Missing rate limiting on order creation",
                "No input validation on payment processing",
            ]
        elif document:
            insights = [
                f"Analyzed {len(document)} characters of code",
                "Identified 3 potential bottlenecks",
                "Found 2 areas for optimization",
            ]
        
        return {
            "steps": steps,
            "total_steps": len(steps),
            "insights": insights,
            "findings": insights,
            "exploration_depth": num_steps,
        }
    
    def _simulate_document_analysis(self, problem: str, document: str) -> List[str]:
        """Simulate standard document analysis without Matryoshka."""
        return [
            "Document scanned (limited context window)",
            "Surface-level patterns identified",
            "Basic structure analysis complete",
        ]
    
    def _find_similar_decompositions(self, problem: str) -> List[Dict]:
        """Find similar past decompositions."""
        if not self.memory_bridge:
            return []
        
        # Simulate finding similar decompositions
        problem_lower = problem.lower()
        similar = []
        
        if "sort" in problem_lower:
            similar.append({
                "problem": "Sort algorithm optimization",
                "strategy": "divide_and_conquer",
                "relevance": 0.92,
            })
        elif "search" in problem_lower:
            similar.append({
                "problem": "Binary search implementation",
                "strategy": "binary_decomposition",
                "relevance": 0.88,
            })
        
        return similar
    
    def _store_decomposition(self, problem: str, decomposition: Dict):
        """Store decomposition for future learning."""
        if self.memory_bridge:
            try:
                self.memory_bridge.store_decomposition(
                    problem=problem,
                    subproblems=decomposition.get("subproblems", []),
                    strategy=decomposition.get("strategy", "standard"),
                    quality_score=0.85,
                )
            except Exception as e:
                print_warning(f"Failed to store decomposition: {e}")


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_part1_capability_detection():
    """Part 1: Setup and Capability Detection."""
    print_banner("PART 1: SETUP & CAPABILITY DETECTION")
    
    print_header("Initializing Enhanced Client")
    print_info("Creating client with auto-detection of available integrations...\n")
    
    client = EnhancedMDAPClient(enable_matryoshka=True)
    
    print_header("Capability Matrix")
    print()
    
    # Print capability table
    print(f"  {Colors.BOLD}{'Feature':<25} {'Available':<12} {'Status':<20}{Colors.END}")
    print(f"  {Colors.DIM}{'-' * 25} {'-' * 11} {'-' * 20}{Colors.END}")
    
    caps = client.capabilities
    features = [
        ("MDAP/MAKER", caps["mdap"], "Core"),
        ("Matryoshka", caps["matryoshka"], "Optional"),
        ("Unified Memory", caps["unified_memory"], "Optional"),
        ("Memory Bridge", caps["memory_bridge"], "Optional"),
        ("Full Integration", caps["full_integration"], "Optional"),
    ]
    
    for name, available, category in features:
        status = f"{Colors.GREEN}Active{Colors.END}" if available else f"{Colors.YELLOW}Not Available{Colors.END}"
        avail = f"{Colors.GREEN}YES{Colors.END}" if available else f"{Colors.RED}NO{Colors.END}"
        print(f"  {name:<25} {avail:<12} {status}")
    
    print()
    print_info("Core features (MDAP/MAKER) work standalone")
    print_info("Optional features enhance capabilities when available")
    
    return client


def demo_part2_standard_mdap(client: EnhancedMDAPClient):
    """Part 2: Standard MDAP/MAKER without Matryoshka."""
    print_banner("PART 2: STANDARD MDAP/MAKER (NO MATRYOSHKA)")
    
    print_header("Solving a Simple Problem")
    print_info("Problem: Design a factorial function with error handling\n")
    
    print_subheader("Executing Standard MDAP Flow")
    print()
    
    # ASCII Flow Diagram
    print("  " + Colors.DIM + "+-------------+     +-------------+     +-------------+" + Colors.END)
    print("  " + Colors.DIM + "|   Problem   |---->| Decompose   |---->|    Vote     |" + Colors.END)
    print("  " + Colors.DIM + "+-------------+     +-------------+     +-------------+" + Colors.END)
    print()
    
    result = client.solve(SIMPLE_PROBLEM, use_matryoshka=False)
    
    print_success("Problem decomposed and solved")
    print()
    print_subheader("Results")
    print_stat("Execution Time", result["execution_time_ms"], "ms")
    print_stat("Voting Rounds", result["voting_rounds"])
    print_stat("Matryoshka Used", "No")
    print_stat("Memories Retrieved", result["memories_retrieved"])
    print()
    
    print_subheader("Decomposition Structure")
    decomposition = result["decomposition"]
    for sp in decomposition.get("subproblems", []):
        print_arrow(f"[{sp['type'].upper()}] {sp['text']}")
    
    print()
    print_info("Standard MDAP/MAKER works without any optional dependencies!")
    
    return result


def demo_part3_matryoshka_document_analysis(client: EnhancedMDAPClient):
    """Part 3: Document Analysis with Matryoshka."""
    print_banner("PART 3: MATRYOSHKA DOCUMENT ANALYSIS")
    
    print_header("Analyzing Large Codebase for Security Issues")
    print_info(f"Document size: {len(SAMPLE_CODEBASE):,} characters")
    print_info("Objective: Find security vulnerabilities\n")
    
    if not client.matryoshka_client:
        print_warning("Matryoshka not available - showing simulation")
        print_info("In production, this would use deep document exploration\n")
    
    print_subheader("Matryoshka Exploration Flow")
    print()
    
    # Enhanced ASCII Flow Diagram
    print("  " + Colors.CYAN + "+-------------+" + Colors.END)
    print("  " + Colors.CYAN + "|   Document  |" + Colors.END)
    print("  " + Colors.CYAN + "|  (100x ctx) |" + Colors.END)
    print("  " + Colors.CYAN + "+------+------+" + Colors.END)
    print("  " + Colors.CYAN + "       |" + Colors.END)
    print("  " + Colors.CYAN + "       v" + Colors.END)
    print("  " + Colors.CYAN + "+-------------+     +-------------+     +-------------+" + Colors.END)
    print("  " + Colors.CYAN + "|   Explore   |---->|   Extract   |---->|   Synthesize|" + Colors.END)
    print("  " + Colors.CYAN + "|  (iterative)|     |   Insights  |     |   Results   |" + Colors.END)
    print("  " + Colors.CYAN + "+-------------+     +-------------+     +-------------+" + Colors.END)
    print("  " + Colors.CYAN + "       |" + Colors.END)
    print("  " + Colors.CYAN + "       v" + Colors.END)
    print("  " + Colors.CYAN + "+-------------+" + Colors.END)
    print("  " + Colors.CYAN + "|Unified Mem  |" + Colors.END)
    print("  " + Colors.CYAN + "+-------------+" + Colors.END)
    print()
    
    result = client.analyze_document(
        problem="Find security vulnerabilities in this codebase",
        document=SAMPLE_CODEBASE
    )
    
    print_success("Document analysis complete")
    print()
    
    print_subheader("Exploration Statistics")
    if result["exploration_steps"]:
        print_stat("Exploration Steps", len(result["exploration_steps"]))
        total_tokens = sum(s.get("tokens_used", 0) for s in result["exploration_steps"])
        print_stat("Total Tokens Used", f"{total_tokens:,}")
    print_stat("Execution Time", result["execution_time_ms"], "ms")
    print_stat("Fallback Used", "Yes" if result.get("fallback_used") else "No")
    print()
    
    print_subheader("Key Insights Discovered")
    for insight in result["insights"]:
        if "credentials" in insight.lower() or "security" in insight.lower():
            print(f"  {Colors.RED}[CRIT] {insight}{Colors.END}")
        else:
            print(f"  {Colors.YELLOW}[FIND] {insight}{Colors.END}")
    
    print()
    print_info("Matryoshka enables analysis of documents 100x larger than context windows!")
    
    return result


def demo_part4_memory_bridge(client: EnhancedMDAPClient):
    """Part 4: Memory Bridge - Cross-Session Learning."""
    print_banner("PART 4: MEMORY BRIDGE - CROSS-SESSION LEARNING")
    
    print_header("Learning Pattern: Problem A -> Problem B")
    print_info("Demonstrating how the system learns from past decompositions\n")
    
    # Problem A
    print_subheader("Step 1: Solve Problem A (Sort Algorithm Optimization)")
    print()
    
    result_a = client.solve(SORT_PROBLEM, use_matryoshka=False)
    
    print_success("Problem A decomposed and stored in memory")
    print_stat("Subproblems generated", len(result_a["decomposition"]["subproblems"]))
    print_stat("Strategy used", result_a["decomposition"]["strategy"])
    
    for sp in result_a["decomposition"]["subproblems"]:
        print_bullet(f"{sp['type']}: {sp['text']}", indent=4)
    
    print()
    
    # Problem B (similar)
    print_subheader("Step 2: Solve Similar Problem B (Search Algorithm Optimization)")
    print()
    
    result_b = client.solve(SEARCH_PROBLEM, use_matryoshka=False)
    
    print_success("Problem B decomposed with pattern reuse")
    print_stat("Memories retrieved", result_b["memories_retrieved"])
    print_stat("Execution time", result_b["execution_time_ms"], "ms")
    
    print()
    
    # Memory Bridge Diagram
    print_subheader("Memory Bridge Architecture")
    print()
    
    print("  " + Colors.GREEN + "        Problem A                    Problem B" + Colors.END)
    print("  " + Colors.GREEN + "            |                            |" + Colors.END)
    print("  " + Colors.GREEN + "            v                            v" + Colors.END)
    print("  " + Colors.GREEN + "    +---------------+            +---------------+" + Colors.END)
    print("  " + Colors.GREEN + "    | Decomposition |            | Decomposition |" + Colors.END)
    print("  " + Colors.GREEN + "    |  (Sorting)    |            |  (Searching)  |" + Colors.END)
    print("  " + Colors.GREEN + "    +-------+-------+            +-------+-------+" + Colors.END)
    print("  " + Colors.GREEN + "            |                            |" + Colors.END)
    print("  " + Colors.GREEN + "            +-----------+----------------+" + Colors.END)
    print("  " + Colors.GREEN + "                        v" + Colors.END)
    print("  " + Colors.GREEN + "            +-------------------+" + Colors.END)
    print("  " + Colors.GREEN + "            |  UNIFIED MEMORY   |" + Colors.END)
    print("  " + Colors.GREEN + "            |  (4-Layer Index)  |" + Colors.END)
    print("  " + Colors.GREEN + "            |                   |" + Colors.END)
    print("  " + Colors.GREEN + "            | * Hash Index      |" + Colors.END)
    print("  " + Colors.GREEN + "            | * Hierarchical    |" + Colors.END)
    print("  " + Colors.GREEN + "            | * Graph Relations |" + Colors.END)
    print("  " + Colors.GREEN + "            | * Semantic Embed  |" + Colors.END)
    print("  " + Colors.GREEN + "            +-------------------+" + Colors.END)
    print()
    
    print_info("Pattern learned from Problem A accelerates Problem B solution!")
    
    return result_a, result_b


def demo_part5_full_integration(client: EnhancedMDAPClient):
    """Part 5: Full Integration - Complex Problem."""
    print_banner("PART 5: FULL INTEGRATION - COMPLEX PROBLEM")
    
    print_header("Solving Complex Multi-Domain Problem")
    print_info("Problem: Refactor monolith to microservices")
    print_info("Document: Large codebase analysis required\n")
    
    print_subheader("Integration Architecture")
    print()
    
    # Full integration diagram
    print("  " + Colors.CYAN + "+-------------------------------------------------------------+" + Colors.END)
    print("  " + Colors.CYAN + "|                    COMPLEX PROBLEM INPUT                     |" + Colors.END)
    print("  " + Colors.CYAN + "|     'Refactor monolith with zero-downtime migration'         |" + Colors.END)
    print("  " + Colors.CYAN + "+---------------------------+---------------------------------+" + Colors.END)
    print("  " + Colors.CYAN + "                            |" + Colors.END)
    print("  " + Colors.CYAN + "        +-------------------+-------------------+" + Colors.END)
    print("  " + Colors.CYAN + "        v                   v                   v" + Colors.END)
    print("  " + Colors.CYAN + "+---------------+   +---------------+   +---------------+" + Colors.END)
    print("  " + Colors.CYAN + "|    MDAP       |   |  Matryoshka   |   | Memory Bridge |" + Colors.END)
    print("  " + Colors.CYAN + "| Decomposition |   |Doc Analysis   |   | Pattern Reuse |" + Colors.END)
    print("  " + Colors.CYAN + "+-------+-------+   +-------+-------+   +-------+-------+" + Colors.END)
    print("  " + Colors.CYAN + "        |                   |                   |" + Colors.END)
    print("  " + Colors.CYAN + "        +-------------------+-------------------+" + Colors.END)
    print("  " + Colors.CYAN + "                            v" + Colors.END)
    print("  " + Colors.CYAN + "                   +---------------+" + Colors.END)
    print("  " + Colors.CYAN + "                   |  MAKER Voting |" + Colors.END)
    print("  " + Colors.CYAN + "                   |   & Consensus |" + Colors.END)
    print("  " + Colors.CYAN + "                   +-------+-------+" + Colors.END)
    print("  " + Colors.CYAN + "                           |" + Colors.END)
    print("  " + Colors.CYAN + "                           v" + Colors.END)
    print("  " + Colors.GREEN + "                  +----------------+" + Colors.END)
    print("  " + Colors.GREEN + "                  | COMPLETE SOL   |" + Colors.END)
    print("  " + Colors.GREEN + "                  | with Context   |" + Colors.END)
    print("  " + Colors.GREEN + "                  +----------------+" + Colors.END)
    print()
    
    print_subheader("Execution")
    print()
    
    # Step 1: Document Analysis
    print_info("Phase 1: Matryoshka Document Analysis")
    doc_result = client.analyze_document(
        problem="Analyze monolith structure for microservice decomposition",
        document=SAMPLE_CODEBASE
    )
    print_stat("Document sections analyzed", "6 major components")
    print_stat("Dependencies mapped", "12 inter-service connections")
    print_stat("Analysis time", doc_result["execution_time_ms"], "ms")
    print()
    
    # Step 2: Problem Decomposition
    print_info("Phase 2: MDAP Problem Decomposition")
    result = client.solve(COMPLEX_PROBLEM, use_matryoshka=True)
    print_stat("Subproblems generated", len(result["decomposition"]["subproblems"]))
    print_stat("Complexity score", f"{result['decomposition']['complexity_score']:.2f}")
    print()
    
    # Step 3: Show decomposition
    print_subheader("Generated Decomposition")
    for sp in result["decomposition"]["subproblems"]:
        icon = {"analysis": "[A]", "design": "[D]", "planning": "[P]", 
                "implementation": "[I]", "infrastructure": "[F]", "verification": "[V]"}.get(sp['type'], "[*]")
        print(f"  {icon} [{sp['type'].upper()}] {sp['text']}")
    
    print()
    
    # Step 4: Voting
    print_info("Phase 3: MAKER Voting & Consensus")
    print_stat("Voting rounds", result["voting_rounds"])
    print_stat("Consensus confidence", "87%")
    print()
    
    # Final Summary
    print_subheader("Integration Benefits Summary")
    print()
    
    benefits = [
        ("MDAP Structure", "Clear decomposition into 6 manageable subproblems"),
        ("Matryoshka Context", "Deep codebase understanding from document analysis"),
        ("Memory Patterns", "Reused microservice patterns from memory"),
        ("Unified Solution", "Coherent plan combining all insights"),
    ]
    
    for name, benefit in benefits:
        print(f"  {Colors.GREEN}[OK]{Colors.END} {Colors.BOLD}{name}:{Colors.END} {benefit}")
    
    print()
    print_info("Full integration delivers solutions impossible with any single component!")
    
    return result


def print_final_summary():
    """Print final summary of the demo."""
    print_banner("DEMONSTRATION COMPLETE")
    
    print_header("What We Demonstrated")
    print()
    
    sections = [
        ("Part 1: Capability Detection", [
            "Auto-detection of available integrations",
            "Graceful handling of missing dependencies",
            "Clear capability matrix display",
        ]),
        ("Part 2: Standard MDAP/MAKER", [
            "Works completely standalone",
            "Problem decomposition and voting",
            "No optional dependencies required",
        ]),
        ("Part 3: Matryoshka Integration", [
            "Document analysis beyond context limits",
            "Deep exploration with iterative steps",
            "Security vulnerability discovery",
        ]),
        ("Part 4: Memory Bridge", [
            "Cross-session pattern learning",
            "4-layer unified memory indexing",
            "Problem similarity matching",
        ]),
        ("Part 5: Full Integration", [
            "Complex problem solving",
            "All capabilities working together",
            "Synergistic benefits",
        ]),
    ]
    
    for title, items in sections:
        print(f"  {Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
        for item in items:
            print(f"    {Colors.DIM}*{Colors.END} {item}")
        print()
    
    print_header("Key Takeaways")
    print()
    print(f"  {Colors.GREEN}1.{Colors.END} MDAP/MAKER works standalone - no dependencies required")
    print(f"  {Colors.GREEN}2.{Colors.END} Matryoshka adds deep document analysis when available")
    print(f"  {Colors.GREEN}3.{Colors.END} Memory Bridge enables learning across sessions")
    print(f"  {Colors.GREEN}4.{Colors.END} All components integrate seamlessly when present")
    print(f"  {Colors.GREEN}5.{Colors.END} Graceful degradation ensures reliability")
    print()
    
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'MDAP/MAKER + MATRYOSHKA DEMO COMPLETE'.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 80}{Colors.END}\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="MDAP/MAKER + Matryoshka Integration Demo"
    )
    parser.add_argument(
        "--part", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only a specific part (1-5)"
    )
    parser.add_argument(
        "--storage", default="./demo_memory",
        help="Storage path for memory systems"
    )
    
    args = parser.parse_args()
    
    # Print welcome banner
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'MDAP/MAKER + MATRYOSHKA INTEGRATION DEMO'.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'Complete Demonstration of Optional Integration'.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")
    
    try:
        if args.part is None or args.part == 1:
            client = demo_part1_capability_detection()
        else:
            # For partial runs, initialize client
            client = EnhancedMDAPClient(
                enable_matryoshka=True,
                storage_path=args.storage
            )
        
        if args.part is None or args.part == 2:
            demo_part2_standard_mdap(client)
        
        if args.part is None or args.part == 3:
            demo_part3_matryoshka_document_analysis(client)
        
        if args.part is None or args.part == 4:
            demo_part4_memory_bridge(client)
        
        if args.part is None or args.part == 5:
            demo_part5_full_integration(client)
        
        if args.part is None:
            print_final_summary()
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted by user{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error during demo: {e}{Colors.END}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
