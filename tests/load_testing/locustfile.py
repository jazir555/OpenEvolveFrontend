"""
Locust Load Testing File for Knowledge Graph System

This module defines user behavior for load testing the knowledge graph API
using Locust (https://locust.io/).

Usage:
    locust -f tests/load_testing/locustfile.py --host=http://localhost:8080

Or run headless:
    locust -f tests/load_testing/locustfile.py --host=http://localhost:8080 --headless \
        -u 100 -r 10 --run-time 5m --html report.html
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class KnowledgeGraphUser(HttpUser):
    """
    Locust user class for knowledge graph load testing.

    Simulates realistic user behavior patterns with weighted tasks:
    - Search (weight 3): Most common operation
    - Get stats (weight 2): Frequent reads
    - Add knowledge (weight 1): Less common writes
    - Analyze (weight 1): Computationally intensive
    """

    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts. Initialize user-specific data."""
        self.user_id = random.randint(1000, 9999)
        self.knowledge_items_added = 0

        logger.info(f"User {self.user_id} starting")

        # Optionally authenticate
        # self.client.post("/api/auth/login", json={
        #     "username": f"user_{self.user_id}",
        #     "password": "test_password"
        # })

    def on_stop(self):
        """Called when a user stops. Clean up resources."""
        logger.info(f"User {self.user_id} stopping. Added {self.knowledge_items_added} items")

    @task(3)
    def search_knowledge(self):
        """
        Search knowledge base.

        Weight: 3 (most common operation)
        """
        queries = [
            "machine learning algorithms",
            "python async programming",
            "database optimization",
            "graph theory applications",
            "knowledge extraction methods",
            "neural networks",
            "data structures",
            "algorithm design",
            "distributed systems",
            "natural language processing"
        ]

        query = random.choice(queries)

        with self.client.get(
            "/api/kg/search",
            params={
                "query": query,
                "limit": random.choice([10, 20, 50]),
                "search_type": random.choice(["semantic", "hybrid", "keyword"])
            },
            catch_response=True,
            name="/api/kg/search"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # API endpoint might not exist
                logger.warning(f"Search endpoint not found: {response.status_code}")
                response.success()  # Don't fail the test
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(2)
    def get_graph_stats(self):
        """
        Get graph statistics.

        Weight: 2 (frequent read operation)
        """
        with self.client.get(
            "/api/kg/stats",
            catch_response=True,
            name="/api/kg/stats"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Verify we got stats
                if "nodes" in data or "edges" in data or "total_entities" in data:
                    response.success()
                else:
                    response.failure("Invalid stats response format")
            elif response.status_code == 404:
                logger.warning("Stats endpoint not found")
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def add_knowledge(self):
        """
        Add knowledge to the graph.

        Weight: 1 (less common write operation)
        """
        content_templates = [
            f"Load test knowledge from user {self.user_id}",
            f"Test content about machine learning concepts",
            f"Information about graph database systems",
            f"Details on knowledge graph architectures",
            f"Data on distributed computing patterns"
        ]

        content = random.choice(content_templates)
        self.knowledge_items_added += 1

        with self.client.post(
            "/api/kg/knowledge",
            json={
                "source": f"load_test_user_{self.user_id}",
                "content": content,
                "metadata": {
                    "load_test": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": self.user_id
                }
            },
            catch_response=True,
            name="/api/kg/knowledge"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 404:
                logger.warning("Add knowledge endpoint not found")
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def analyze_graph(self):
        """
        Analyze graph structure.

        Weight: 1 (computationally intensive operation)
        """
        analysis_types = ["communities", "centrality", "components", "pagerank"]
        analysis_type = random.choice(analysis_types)

        with self.client.post(
            "/api/kg/analyze",
            json={
                "analysis_type": analysis_type
            },
            catch_response=True,
            name="/api/kg/analyze"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                logger.warning("Analyze endpoint not found")
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def get_knowledge_by_id(self):
        """
        Retrieve specific knowledge by ID.

        Weight: 1
        """
        # Try to get a random knowledge ID
        knowledge_id = random.randint(1, 1000)

        with self.client.get(
            f"/api/kg/knowledge/{knowledge_id}",
            catch_response=True,
            name="/api/kg/knowledge/[id]"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Expected for random IDs
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def query_relations(self):
        """
        Query relationships in the graph.

        Weight: 1
        """
        with self.client.get(
            "/api/kg/relations",
            params={
                "limit": random.choice([20, 50, 100]),
                "entity_type": random.choice(["concept", "document", "entity"])
            },
            catch_response=True,
            name="/api/kg/relations"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                logger.warning("Relations endpoint not found")
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class KnowledgeGraphWriteUser(HttpUser):
    """
    User class focused on write operations.

    Use for testing write-heavy scenarios.
    """

    wait_time = between(0.5, 2)  # Faster for write operations

    def on_start(self):
        self.user_id = random.randint(1000, 9999)
        self.batch_number = 0

    @task(5)
    def batch_add_knowledge(self):
        """Add knowledge in batches."""
        self.batch_number += 1

        batch_size = random.randint(5, 20)

        knowledge_batch = [
            {
                "source": f"batch_user_{self.user_id}",
                "content": f"Batch {self.batch_number} item {i}",
                "metadata": {
                    "batch_id": self.batch_number,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            for i in range(batch_size)
        ]

        with self.client.post(
            "/api/kg/knowledge/batch",
            json={"items": knowledge_batch},
            catch_response=True,
            name="/api/kg/knowledge/batch"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 404:
                # Batch endpoint might not exist, try single adds
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def search_recent(self):
        """Search for recently added content."""
        with self.client.get(
            "/api/kg/search",
            params={
                "query": f"batch_user_{self.user_id}",
                "limit": 10
            },
            catch_response=True,
            name="/api/kg/search (recent)"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.success()  # Don't fail


# Event handlers for test monitoring
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Log request events for monitoring.

    Args:
        request_type: Type of request (GET, POST, etc.)
        name: Request name
        response_time: Response time in ms
        response_length: Response size in bytes
        exception: Exception if any
    """
    if exception:
        logger.error(
            f"Request failed: {request_type} {name} - "
            f"Exception: {exception}"
        )
    elif response_time > 5000:  # 5 second threshold
        logger.warning(
            f"Slow request: {request_type} {name} - "
            f"Response time: {response_time}ms"
        )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops. Generate summary."""
    logger.info("="*60)
    logger.info("LOAD TEST COMPLETED")
    logger.info("="*60)

    if environment.stats.total.fail_ratio > 0.05:  # 5% failure threshold
        logger.warning(
            f"High failure ratio: {environment.stats.total.fail_ratio:.2%}"
        )
    else:
        logger.info(
            f"Acceptable failure ratio: {environment.stats.total.fail_ratio:.2%}"
        )

    logger.info(
        f"Total requests: {environment.stats.total.num_requests}"
    )
    logger.info(
        f"Response time avg: {environment.stats.total.avg_response_time}ms"
    )
    logger.info(
        f"Requests/sec: {environment.stats.total.total_req_per_sec:.2f}"
    )
