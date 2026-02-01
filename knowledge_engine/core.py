from typing import Dict, Any, List, Optional

class KnowledgeState:
    """
    Represents the current state of knowledge for a given query or context.
    """
    def __init__(self, query: str):
        self.query: str = query
        self.facts: List[str] = []
        self.uncertainties: List[str] = []
        self.search_history: List[Dict[str, Any]] = []
        self.candidate_answers: List[str] = []
        self.current_understanding: str = ""

    def add_fact(self, fact: str):
        self.facts.append(fact)

    def add_uncertainty(self, uncertainty: str):
        self.uncertainties.append(uncertainty)

    def add_search_result(self, search_result: Dict[str, Any]):
        self.search_history.append(search_result)

    def set_current_understanding(self, understanding: str):
        self.current_understanding = understanding

    def add_workflow_execution(self, workflow_id: str, artifacts_extracted: int, timestamp: str):
        """Record a workflow execution in search history."""
        self.search_history.append({
            "workflow_id": workflow_id,
            "artifacts_extracted": artifacts_extracted,
            "timestamp": timestamp
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "facts": self.facts,
            "uncertainties": self.uncertainties,
            "search_history": self.search_history,
            "candidate_answers": self.candidate_answers,
            "current_understanding": self.current_understanding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeState':
        state = cls(data["query"])
        state.facts = data.get("facts", [])
        state.uncertainties = data.get("uncertainties", [])
        state.search_history = data.get("search_history", [])
        state.candidate_answers = data.get("candidate_answers", [])
        state.current_understanding = data.get("current_understanding", "")
        return state

class EntityKnowledgeGraph:
    """
    A simple in-memory representation of an entity knowledge graph.
    Nodes are entities, edges represent relationships.
    """
    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []

    def get_entities(self) -> List[str]:
        """Return list of all entity names."""
        return list(self.entities.keys())

    def search_entities(self, query: str) -> List[str]:
        """Search for entities matching the query."""
        query_lower = query.lower()
        return [name for name in self.entities.keys() if query_lower in name.lower()]

    def add_entity(self, entity_name: str, attributes: Optional[Dict[str, Any]] = None):
        if entity_name not in self.entities:
            self.entities[entity_name] = attributes if attributes is not None else {}

    async def add_entity_async(self, entity_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Async version of add_entity for compatibility with async tests."""
        self.add_entity(entity_name, attributes)

    def add_relationship(self, entity1: str, relation: str, entity2: str, attributes: Optional[Dict[str, Any]] = None):
        self.add_entity(entity1)
        self.add_entity(entity2)
        relationship = {
            "source": entity1,
            "relation": relation,
            "target": entity2,
            "attributes": attributes if attributes is not None else {}
        }
        self.relationships.append(relationship)

    async def add_relationship_async(self, entity1: str, relation: str, entity2: str, attributes: Optional[Dict[str, Any]] = None):
        """Async version of add_relationship for compatibility with async tests."""
        self.add_relationship(entity1, relation, entity2, attributes)

    def add_decision_link(self, entity_name: str, decision_id: str):
        """Link an entity to an ADR decision record."""
        self.add_entity(entity_name)
        entity = self.entities.get(entity_name, {})
        entity["DECIDED_BY"] = decision_id
        self.entities[entity_name] = entity
        self.add_relationship(entity_name, "DECIDED_BY", decision_id)

    def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_name)

    def get_relationships_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        return [rel for rel in self.relationships if rel["source"] == entity_name or rel["target"] == entity_name]

    async def get_relationships_for_entity_async(self, entity_name: str) -> List[Dict[str, Any]]:
        """Async version for compatibility with async tests."""
        return self.get_relationships_for_entity(entity_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "relationships": self.relationships,
        }

    async def to_dict_async(self) -> Dict[str, Any]:
        """Async version for compatibility with async tests."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityKnowledgeGraph':
        graph = cls()
        graph.entities = data.get("entities", {})
        graph.relationships = data.get("relationships", [])
        return graph
