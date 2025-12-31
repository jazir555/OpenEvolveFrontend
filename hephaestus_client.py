import requests
import json
from typing import Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)

class HephaestusClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request to {endpoint} failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def create_ticket(self, title: str, description: str, workflow_id: str) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/tickets/create"
        payload = {"title": title, "description": description, "workflow_id": workflow_id}
        return self._request("POST", endpoint, json=payload, headers=self.headers)

    def update_ticket_dependencies(self, ticket_id: str, blocked_by_ids: List[str]) -> Dict[str, Any]:
        """
        Update ticket dependencies by updating the blocked_by_ticket_ids field.
        This method uses the existing /tickets/update endpoint in Hephaestus
        to update the blocked_by_ticket_ids field of a ticket.
        """
        endpoint = f"{self.base_url}/tickets/update"
        payload = {
            "ticket_id": ticket_id,
            "updates": {
                "blocked_by_ticket_ids": blocked_by_ids
            }
        }
        return self._request("POST", endpoint, json=payload, headers=self.headers)

    def get_workflow_tickets(self, workflow_id: str) -> List[Dict[str, Any]]:
        endpoint = f"{self.base_url}/workflows/{workflow_id}/tickets"
        return self._request("GET", endpoint, headers=self.headers)

    def create_workflow(self, problem_statement: str, content_analyzer_team: str, planner_team: str, solver_team: str, patcher_team: str, assembler_team: str, sub_problem_red_gauntlet: str, sub_problem_gold_gauntlet: str, final_red_gauntlet: str, final_gold_gauntlet: str, solver_generation_gauntlet: str) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/workflows"
        payload = {
            "problem_statement": problem_statement,
            "content_analyzer_team": content_analyzer_team,
            "planner_team": planner_team,
            "solver_team": solver_team,
            "patcher_team": patcher_team,
            "assembler_team": assembler_team,
            "sub_problem_red_gauntlet": sub_problem_red_gauntlet,
            "sub_problem_gold_gauntlet": sub_problem_gold_gauntlet,
            "final_red_gauntlet": final_red_gauntlet,
            "final_gold_gauntlet": final_gold_gauntlet,
            "solver_generation_gauntlet": solver_generation_gauntlet,
        }
        return self._request("POST", endpoint, json=payload, headers=self.headers)

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/workflows/{workflow_id}"
        return self._request("GET", endpoint, headers=self.headers)
