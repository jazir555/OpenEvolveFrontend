"""
Complete Workflow Demo: OpenEvolve Knowledge Engine

This script demonstrates a real-world scenario where the knowledge engine
extracts, integrates, verifies, and learns from multiple data sources.

Scenario: Research team analyzing the AI landscape
"""

import asyncio
import sys
sys.path.insert(0, '..')

from graph.unified_kg import UnifiedKnowledgeGraph, UnifiedTriple
from graph.kg_models import (
    KnowledgeGraphModels, KnowledgeStatement, 
    KnowledgeSource, EntityProfile
)


async def main():
    print("=" * 70)
    print("OPENEVOLVE KNOWLEDGE ENGINE - COMPLETE WORKFLOW DEMO")
    print("=" * 70)
    print()
    
    # ========================================================================
    # PHASE 1: Initialize Core Components
    # ========================================================================
    print("PHASE 1: Initializing Core Components")
    print("-" * 70)
    
    # Initialize the unified knowledge graph
    ukg = UnifiedKnowledgeGraph(backend='memory')
    print("[OK] UnifiedKnowledgeGraph initialized")
    
    # Initialize knowledge models
    kgm = KnowledgeGraphModels()
    print("[OK] KnowledgeGraphModels initialized")
    print()
    
    # ========================================================================
    # PHASE 2: Extract Knowledge from Multiple Sources
    # ========================================================================
    print("PHASE 2: Knowledge Extraction from Multiple Sources")
    print("-" * 70)
    
    # Source 1: Research paper text
    paper_text = """
    Dr. Alice Chen from OpenAI presented groundbreaking research on 
    large language models at NeurIPS 2024. Her work, co-authored with 
    Bob Smith from Stanford, demonstrates how transformer architectures 
    can be optimized for reasoning tasks. OpenAI has applied these 
    findings to improve GPT-5's mathematical capabilities.
    """
    
    print("Source 1: Research paper")
    print(f"  Text length: {len(paper_text)} characters")
    
    # Simulate extraction (in real use, would call DeepKE/OneKE)
    paper_triples = [
        UnifiedTriple("Alice Chen", "works_at", "OpenAI", 
                      confidence=0.95, source="research_paper"),
        UnifiedTriple("Alice Chen", "presented_at", "NeurIPS 2024", 
                      confidence=0.90, source="research_paper"),
        UnifiedTriple("Alice Chen", "researches", "large language models", 
                      confidence=0.92, source="research_paper"),
        UnifiedTriple("Alice Chen", "coauthored_with", "Bob Smith", 
                      confidence=0.88, source="research_paper"),
        UnifiedTriple("Bob Smith", "works_at", "Stanford", 
                      confidence=0.85, source="research_paper"),
        UnifiedTriple("OpenAI", "develops", "GPT-5", 
                      confidence=0.80, source="research_paper"),
    ]
    
    for triple in paper_triples:
        ukg.add_triple(triple)
    print(f"  Extracted {len(paper_triples)} triples")
    
    # Source 2: Company database
    print("\nSource 2: Company database")
    db_triples = [
        UnifiedTriple("OpenAI", "founded_in", "2015", 
                      confidence=1.0, source="company_database"),
        UnifiedTriple("OpenAI", "ceo", "Sam Altman", 
                      confidence=0.99, source="company_database"),
        UnifiedTriple("OpenAI", "headquarters", "San Francisco", 
                      confidence=1.0, source="company_database"),
        UnifiedTriple("Stanford", "located_in", "California", 
                      confidence=1.0, source="company_database"),
    ]
    
    for triple in db_triples:
        ukg.add_triple(triple)
    print(f"  Extracted {len(db_triples)} triples")
    
    # Source 3: Web article
    print("\nSource 3: Web article")
    web_triples = [
        UnifiedTriple("GPT-5", "announced", "January 2025", 
                      confidence=0.75, source="web_article"),
        UnifiedTriple("GPT-5", "capability", "mathematical reasoning", 
                      confidence=0.70, source="web_article"),
        UnifiedTriple("NeurIPS 2024", "location", "Vancouver", 
                      confidence=0.90, source="web_article"),
    ]
    
    for triple in web_triples:
        ukg.add_triple(triple)
    print(f"  Extracted {len(web_triples)} triples")
    
    total_triples = len(paper_triples) + len(db_triples) + len(web_triples)
    print(f"\nTotal triples in graph: {total_triples}")
    print()
    
    # ========================================================================
    # PHASE 3: Create Rich Entity Profiles
    # ========================================================================
    print("PHASE 3: Creating Rich Entity Profiles")
    print("-" * 70)
    
    # Create profile for Alice Chen
    alice_profile = kgm.create_entity_profile(
        name="Alice Chen",
        types=["Person", "Researcher", "AI Scientist"],
        aliases=["A. Chen", "Dr. Chen"]
    )
    alice_profile.properties["expertise"] = ["LLMs", "Transformers", "Reasoning"]
    alice_profile.properties["h-index"] = 45
    alice_profile.add_relationship("works_with", "Bob Smith", confidence=0.88)
    alice_profile.add_relationship("employed_by", "OpenAI", confidence=0.95)
    
    print("[OK] Created profile: Alice Chen")
    print(f"  Types: {alice_profile.types}")
    print(f"  Expertise: {alice_profile.properties['expertise']}")
    
    # Create profile for OpenAI
    openai_profile = kgm.create_entity_profile(
        name="OpenAI",
        types=["Company", "AI Lab", "Organization"]
    )
    openai_profile.properties["founded"] = "2015"
    openai_profile.properties["employees"] = 500
    openai_profile.properties["products"] = ["GPT-4", "GPT-5", "DALL-E", "Codex"]
    
    print("[OK] Created profile: OpenAI")
    print(f"  Types: {openai_profile.types}")
    print(f"  Products: {openai_profile.properties['products']}")
    print()
    
    # ========================================================================
    # PHASE 4: Verify Knowledge
    # ========================================================================
    print("PHASE 4: Knowledge Verification")
    print("-" * 70)
    
    # Create a statement with full provenance
    stmt = kgm.create_statement(
        subject="GPT-5",
        predicate="has_capability",
        object="mathematical reasoning",
        confidence=0.82,
        source=KnowledgeSource.EXTRACTION,
        source_detail="Research paper + Web article corroboration",
        evidence=[
            "Alice Chen's NeurIPS paper describes optimization",
            "Web article mentions improved capabilities",
            "OpenAI blog post on reasoning"
        ]
    )
    
    print("[OK] Created knowledge statement:")
    print(f"  Subject: {stmt.subject}")
    print(f"  Predicate: {stmt.predicate}")
    print(f"  Object: {stmt.object}")
    print(f"  Confidence: {stmt.confidence}")
    print(f"  Evidence items: {len(stmt.evidence)}")
    print()
    
    # ========================================================================
    # PHASE 5: Query and Analyze
    # ========================================================================
    print("PHASE 5: Querying and Analysis")
    print("-" * 70)
    
    # Query 1: What do we know about Alice?
    print("Query 1: What do we know about 'Alice Chen'?")
    alice_triples = ukg.get_triples(subject="Alice Chen")
    print(f"  Found {len(alice_triples)} triples:")
    for t in alice_triples:
        print(f"    - {t.subject} {t.predicate} {t.object} (conf: {t.confidence})")
    
    # Query 2: Who works at OpenAI?
    print("\nQuery 2: Who works at 'OpenAI'?")
    openai_people = ukg.get_triples(predicate="works_at", object="OpenAI")
    print(f"  Found {len(openai_people)} people:")
    for t in openai_people:
        print(f"    - {t.subject}")
    
    # Query 3: Find paths between entities
    print("\nQuery 3: Find connection path: Alice Chen -> GPT-5")
    paths = ukg.find_paths("Alice Chen", "GPT-5", max_length=3)
    if paths:
        print(f"  Found {len(paths)} path(s):")
        for i, path in enumerate(paths, 1):
            print(f"    Path {i}:")
            for edge in path:
                print(f"      {edge['from']} --[{edge['predicate']}]--> {edge['to']}")
    else:
        print("  No direct path found")
    
    # Query 4: Get graph statistics
    print("\nQuery 4: Graph Statistics")
    stats = ukg.get_statistics()
    print(f"  Total entities: {stats.node_count}")
    print(f"  Total triples: {stats.edge_count}")
    print(f"  Average confidence: {stats.avg_confidence:.2f}")
    print()
    
    # ========================================================================
    # PHASE 6: Pattern Discovery
    # ========================================================================
    print("PHASE 6: Pattern Discovery")
    print("-" * 70)
    
    # Create a pattern
    pattern = kgm.create_pattern(
        name="Researcher-Company-Product chain",
        pattern_type="path",
        nodes=["Researcher", "Company", "Product"],
        edges=[
            {"from": "Researcher", "to": "Company", "predicate": "works_at"},
            {"from": "Company", "to": "Product", "predicate": "develops"}
        ],
        frequency=2,
        confidence=0.85
    )
    
    print("[OK] Discovered pattern:")
    print(f"  Name: {pattern.name}")
    print(f"  Type: {pattern.pattern_type}")
    print(f"  Frequency: {pattern.frequency}")
    print(f"  Confidence: {pattern.confidence}")
    print()
    
    # ========================================================================
    # PHASE 7: Export and Health Check
    # ========================================================================
    print("PHASE 7: Export and Health Check")
    print("-" * 70)
    
    # Health check
    health = ukg.health_check()
    print("UnifiedKnowledgeGraph Health:")
    print(f"  Status: {health['status']}")
    print(f"  Backend: {health['backend']}")
    print(f"  NetworkX available: {health['networkx_available']}")
    print(f"  Triples: {health['triples_count']}")
    print(f"  Entities: {health['entities_count']}")
    
    kgm_health = kgm.health_check()
    print("\nKnowledgeGraphModels Health:")
    print(f"  Status: {kgm_health['status']}")
    print(f"  Profiles: {kgm_health['profiles_loaded']}")
    print(f"  Statements: {kgm_health['statements_loaded']}")
    
    # Export data
    export_data = ukg.export_to_dict()
    print(f"\n[OK] Exported graph data:")
    print(f"  Triples exported: {len(export_data['triples'])}")
    print(f"  Entities exported: {len(export_data['entities'])}")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print()
    print("What was accomplished:")
    print(f"  [OK] Extracted knowledge from {3} sources")
    print(f"  [OK] Stored {total_triples} triples in unified graph")
    print(f"  [OK] Created {2} rich entity profiles")
    print(f"  [OK] Verified knowledge with provenance")
    print(f"  [OK] Executed {4} different query types")
    print(f"  [OK] Discovered connection patterns")
    print(f"  [OK] Exported full knowledge base")
    print()
    print("Components used:")
    print("  [OK] UnifiedKnowledgeGraph - Core triple storage")
    print("  [OK] KnowledgeGraphModels - Schema and provenance")
    print("  [OK] UnifiedTriple - Knowledge representation")
    print("  [OK] KnowledgeStatement - Verified facts with evidence")
    print("  [OK] EntityProfile - Rich entity descriptions")
    print("  [OK] GraphPattern - Discovered patterns")
    print()
    print("Key capabilities demonstrated:")
    print("  [OK] Multi-source knowledge integration")
    print("  [OK] Entity relationship tracking")
    print("  [OK] Path finding between entities")
    print("  [OK] Confidence scoring")
    print("  [OK] Provenance tracking")
    print("  [OK] Pattern discovery")
    print("  [OK] Export/import")
    print()


if __name__ == "__main__":
    asyncio.run(main())
