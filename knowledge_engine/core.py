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

    def add_entity(self, entity_name: str, attributes: Optional[Dict[str, Any]] = None):
        if entity_name not in self.entities:
            self.entities[entity_name] = attributes if attributes is not None else {}

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "relationships": self.relationships,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityKnowledgeGraph':
        graph = cls()
        graph.entities = data.get("entities", {})
        graph.relationships = data.get("relationships", [])
        return graph
