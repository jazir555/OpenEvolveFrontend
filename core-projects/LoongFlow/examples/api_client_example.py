#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example client for using the LoongFlow API.

This demonstrates how other services can integrate with LoongFlow
via its HTTP API.
"""

import asyncio
import time
from typing import Optional

import requests


class LoongFlowClient:
    """
    Python client for the LoongFlow API.

    Usage:
        client = LoongFlowClient("http://localhost:8000")
        evolution_id = client.start_evolution(
            name="my-evolution",
            task="Solve the packing problem",
            max_generations=10
        )
        status = client.get_status(evolution_id)
        solution = client.get_solution(evolution_id)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the LoongFlow API server
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> dict:
        """
        Check if the API server is healthy.

        Returns:
            Health status dict
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def start_evolution(
        self,
        name: str,
        task: str,
        max_generations: int = 10,
        population_size: int = 50,
        config: Optional[dict] = None
    ) -> str:
        """
        Start a new evolution run.

        Args:
            name: Unique name for this evolution
            task: Task description
            max_generations: Maximum generations to run
            population_size: Population size
            config: Additional configuration overrides

        Returns:
            Evolution ID string
        """
        payload = {
            "name": name,
            "task": task,
            "max_generations": max_generations,
            "population_size": population_size,
        }

        if config:
            payload["config"] = config

        response = requests.post(
            f"{self.base_url}/api/v1/evolve",
            json=payload
        )
        response.raise_for_status()

        data = response.json()
        return data["evolution_id"]

    def get_status(self, evolution_id: str) -> dict:
        """
        Get the status of an evolution.

        Args:
            evolution_id: Evolution ID

        Returns:
            Status dict with fields:
            - evolution_id: str
            - name: str
            - status: str (PENDING, RUNNING, COMPLETED, FAILED)
            - current_generation: int
            - max_generations: int
            - best_fitness: float
            - created_at: str (ISO-8601)
            - updated_at: str (ISO-8601)
            - error: Optional[str]
        """
        response = requests.get(f"{self.base_url}/api/v1/status/{evolution_id}")
        response.raise_for_status()
        return response.json()

    def get_solution(self, evolution_id: str) -> dict:
        """
        Get the solution from a completed evolution.

        Args:
            evolution_id: Evolution ID

        Returns:
            Solution dict with fields:
            - evolution_id: str
            - name: str
            - solution: str
            - fitness: float
            - generations_completed: int
            - metadata: dict
        """
        response = requests.get(f"{self.base_url}/api/v1/solutions/{evolution_id}")
        response.raise_for_status()
        return response.json()

    def list_evolutions(self, status: Optional[str] = None, limit: int = 100) -> dict:
        """
        List all evolutions, optionally filtered by status.

        Args:
            status: Filter by status (PENDING, RUNNING, COMPLETED, FAILED)
            limit: Maximum number of results

        Returns:
            Dict with 'evolutions' list and 'count'
        """
        params = {"limit": limit}
        if status:
            params["status"] = status

        response = requests.get(
            f"{self.base_url}/api/v1/evolutions",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def delete_evolution(self, evolution_id: str) -> dict:
        """
        Delete a completed or failed evolution.

        Args:
            evolution_id: Evolution ID

        Returns:
            Success message
        """
        response = requests.delete(f"{self.base_url}/api/v1/evolutions/{evolution_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        evolution_id: str,
        check_interval: float = 1.0,
        timeout: float = 300.0,
        callback=None
    ) -> dict:
        """
        Wait for an evolution to complete (or fail).

        Args:
            evolution_id: Evolution ID
            check_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            callback: Optional function called with status on each check

        Returns:
            Final status dict

        Raises:
            TimeoutError: If evolution doesn't complete within timeout
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Evolution {evolution_id} did not complete within {timeout}s")

            status = self.get_status(evolution_id)

            if callback:
                callback(status)

            if status["status"] in ["COMPLETED", "FAILED"]:
                return status

            time.sleep(check_interval)


def example_usage():
    """Example of using the LoongFlow client."""

    # Initialize client
    client = LoongFlowClient("http://localhost:8000")

    # Check health
    print("Checking health...")
    health = client.health_check()
    print(f"Health: {health['status']}")
    print()

    # Start an evolution
    print("Starting evolution...")
    evolution_id = client.start_evolution(
        name="example-evolution",
        task="Solve a simple optimization problem",
        max_generations=5
    )
    print(f"Evolution ID: {evolution_id}")
    print()

    # Wait for completion with progress updates
    print("Waiting for completion...")
    try:
        def progress_callback(status):
            gen = status["current_generation"]
            max_gen = status["max_generations"]
            fitness = status["best_fitness"]
            print(f"  Generation {gen}/{max_gen}, Fitness: {fitness:.3f}")

        final_status = client.wait_for_completion(
            evolution_id,
            check_interval=0.5,
            timeout=30.0,
            callback=progress_callback
        )

        print(f"\nFinal status: {final_status['status']}")
        print()

        # Get the solution
        if final_status["status"] == "COMPLETED":
            print("Retrieving solution...")
            solution = client.get_solution(evolution_id)
            print(f"Solution: {solution['solution'][:100]}...")
            print(f"Fitness: {solution['fitness']}")
            print()

            # Clean up
            print("Cleaning up...")
            client.delete_evolution(evolution_id)
            print("Evolution deleted")

    except TimeoutError as e:
        print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}")


def example_list_all():
    """Example of listing all evolutions."""
    client = LoongFlowClient()

    print("All evolutions:")
    result = client.list_evolutions()
    for evo in result["evolutions"]:
        print(f"  - {evo['name']} ({evo['status']})")
    print(f"Total: {result['count']}")


if __name__ == "__main__":
    print("=" * 60)
    print("LoongFlow API Client Example")
    print("=" * 60)
    print()

    # Run the example
    example_usage()

    print()
    print("=" * 60)
