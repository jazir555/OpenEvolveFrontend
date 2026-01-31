"""
Extended Knowledge Artifact Generation Benchmark

Generates 50+ additional knowledge artifacts from diverse scenarios
including edge cases, cross-domain problems, and real-world use cases.
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from knowledge_engine.input_processor import EnhancedInputProcessor
from knowledge_engine.domain_adapter import DomainAdapter, DomainClassifier, TaskDomain
from knowledge_engine.output_validator import OutputValidator
from knowledge_engine.creative_pipeline import CreativeEnhancer
from knowledge_artifact_extractor import KnowledgeArtifactExtractor


class ExtendedKnowledgeArtifactBenchmark:
    """Extended benchmark for generating 50+ additional knowledge artifacts."""
    
    def __init__(self, output_dir: str = "benchmark_artifacts_extended"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.input_processor = EnhancedInputProcessor()
        self.domain_classifier = DomainClassifier()
        self.output_validator = OutputValidator()
        self.creative_enhancer = CreativeEnhancer()
        
        self.artifacts: List[Dict] = []
        self.extractor = KnowledgeArtifactExtractor(
            artifact_store_path=str(self.output_dir / "artifacts.json")
        )
        
        self.stats = {
            "total_scenarios": 0,
            "artifacts_generated": 0,
            "by_type": {},
            "by_domain": {}
        }
    
    # ==================================================================
    # EXTENDED SCENARIO DEFINITIONS (50+ scenarios)
    # ==================================================================
    
    def get_security_scenarios(self) -> List[Dict]:
        """Security-focused scenarios."""
        return [
            {
                "id": "sec_auth_design",
                "type": "technical",
                "prompt": "Design a zero-trust authentication system for a healthcare API handling PHI data",
                "requirements": {"facts": ["oauth2", "mfa", "hipaa", "audit"], "sections": ["flow", "security", "compliance"]},
                "expected_domain": "technical"
            },
            {
                "id": "sec_threat_model",
                "type": "analytical",
                "prompt": "Create a threat model for a cryptocurrency exchange platform",
                "requirements": {"facts": ["attack_vectors", "mitigations", "risk_levels"], "sections": ["threats", "mitigations", "priority"]},
                "expected_domain": "analytical"
            },
            {
                "id": "sec_incident_response",
                "type": "technical",
                "prompt": "Design an incident response playbook for a ransomware attack on cloud infrastructure",
                "requirements": {"facts": ["containment", "eradication", "recovery"], "sections": ["detection", "response", "postmortem"]},
                "expected_domain": "technical"
            },
            {
                "id": "sec_penetration_test",
                "type": "technical",
                "prompt": "Plan a comprehensive penetration test for a fintech mobile application",
                "requirements": {"facts": ["owasp", "api_testing", "reverse_engineering"], "sections": ["scope", "methodology", "deliverables"]},
                "expected_domain": "technical"
            },
            {
                "id": "sec_compliance_gdpr",
                "type": "analytical",
                "prompt": "Conduct a GDPR compliance audit for a SaaS company processing EU citizen data",
                "requirements": {"facts": ["data_subject_rights", "lawful_basis", "dpo"], "sections": ["gaps", "remediation", "timeline"]},
                "expected_domain": "analytical"
            },
            {
                "id": "sec_api_security",
                "type": "technical",
                "prompt": "Implement API security best practices for a public-facing GraphQL endpoint",
                "requirements": {"facts": ["introspection", "depth_limiting", "cost_analysis"], "sections": ["vulnerabilities", "mitigations", "monitoring"]},
                "expected_domain": "technical"
            },
            {
                "id": "sec_secrets_management",
                "type": "technical",
                "prompt": "Design a secrets management strategy for 100+ microservices across 3 cloud providers",
                "requirements": {"facts": ["vault", "rotation", "least_privilege"], "sections": ["architecture", "operations", "recovery"]},
                "expected_domain": "technical"
            },
            {
                "id": "sec_supply_chain",
                "type": "analytical",
                "prompt": "Assess supply chain security risks for a software vendor using 50+ third-party dependencies",
                "requirements": {"facts": ["sbom", "vulnerability_scanning", "sigstore"], "sections": ["risks", "controls", "monitoring"]},
                "expected_domain": "analytical"
            }
        ]
    
    def get_ml_ai_scenarios(self) -> List[Dict]:
        """Machine Learning and AI scenarios."""
        return [
            {
                "id": "ml_model_deployment",
                "type": "technical",
                "prompt": "Design a production ML deployment pipeline for real-time recommendation system with A/B testing",
                "requirements": {"facts": ["feature_store", "model_registry", "canary"], "sections": ["pipeline", "monitoring", "rollback"]},
                "expected_domain": "technical"
            },
            {
                "id": "ml_data_quality",
                "type": "analytical",
                "prompt": "Implement data quality monitoring for a computer vision training pipeline processing 1M images daily",
                "requirements": {"facts": ["drift_detection", "anomalies", "lineage"], "sections": ["metrics", "alerts", "remediation"]},
                "expected_domain": "analytical"
            },
            {
                "id": "ml_explainability",
                "type": "educational",
                "prompt": "Explain how SHAP values work for model interpretability to non-technical stakeholders",
                "requirements": {"audience": "beginner", "facts": ["feature_importance", "baseline", "contributions"]},
                "expected_domain": "educational"
            },
            {
                "id": "ml_llm_prompting",
                "type": "technical",
                "prompt": "Design a prompt engineering framework for consistent JSON output from LLMs",
                "requirements": {"facts": ["json_schema", "few_shot", "validation"], "sections": ["patterns", "testing", "fallbacks"]},
                "expected_domain": "technical"
            },
            {
                "id": "ml_edge_deployment",
                "type": "technical",
                "prompt": "Optimize a TensorFlow model for edge deployment on Raspberry Pi with <2GB RAM",
                "requirements": {"facts": ["quantization", "pruning", "tflite"], "sections": ["optimization", "benchmarking", "tradeoffs"]},
                "expected_domain": "technical"
            },
            {
                "id": "ml_federated_learning",
                "type": "educational",
                "prompt": "Teach federated learning concepts to ML engineers with practical PyTorch examples",
                "requirements": {"audience": "intermediate", "code_blocks": True, "facts": ["privacy", "aggregation", "differential_privacy"]},
                "expected_domain": "educational"
            },
            {
                "id": "ml_bias_detection",
                "type": "analytical",
                "prompt": "Audit a hiring algorithm for demographic bias using fairness metrics",
                "requirements": {"facts": ["disparate_impact", "equalized_odds", "demographic_parity"], "sections": ["metrics", "findings", "recommendations"]},
                "expected_domain": "analytical"
            },
            {
                "id": "ml_vector_database",
                "type": "technical",
                "prompt": "Design a vector database architecture for semantic search across 10M documents with sub-100ms latency",
                "requirements": {"facts": ["embeddings", "ann", "indexing"], "sections": ["architecture", "partitioning", "scaling"]},
                "expected_domain": "technical"
            }
        ]
    
    def get_data_engineering_scenarios(self) -> List[Dict]:
        """Data engineering scenarios."""
        return [
            {
                "id": "data_lake_design",
                "type": "technical",
                "prompt": "Design a data lake architecture for a multi-tenant SaaS with data isolation requirements",
                "requirements": {"facts": ["delta_lake", "iceberg", "governance"], "sections": ["storage", "processing", "access_control"]},
                "expected_domain": "technical"
            },
            {
                "id": "data_streaming_etl",
                "type": "technical",
                "prompt": "Build a real-time ETL pipeline processing 100K events/second with exactly-once semantics",
                "requirements": {"facts": ["kafka", "spark_streaming", "checkpointing"], "sections": ["ingestion", "transformation", "delivery"]},
                "expected_domain": "technical"
            },
            {
                "id": "data_governance",
                "type": "analytical",
                "prompt": "Implement a data governance framework for an enterprise with 500+ data sources",
                "requirements": {"facts": ["lineage", "catalog", "quality"], "sections": ["policies", "implementation", "metrics"]},
                "expected_domain": "analytical"
            },
            {
                "id": "data_migration_cloud",
                "type": "technical",
                "prompt": "Plan a zero-downtime migration of 50TB on-premise data warehouse to Snowflake",
                "requirements": {"facts": ["cdc", "validation", "rollback"], "sections": ["phases", "risks", "validation"]},
                "expected_domain": "technical"
            },
            {
                "id": "data_privacy_pii",
                "type": "technical",
                "prompt": "Implement PII detection and anonymization for a data platform handling customer data",
                "requirements": {"facts": ["ner", "tokenization", "k_anonymity"], "sections": ["detection", "anonymization", "audit"]},
                "expected_domain": "technical"
            },
            {
                "id": "data_observability",
                "type": "analytical",
                "prompt": "Design data observability for a data mesh architecture with 50+ data products",
                "requirements": {"facts": ["data_freshness", "schema_drift", "volume_anomalies"], "sections": ["metrics", "alerting", "remediation"]},
                "expected_domain": "analytical"
            }
        ]
    
    def get_infrastructure_scenarios(self) -> List[Dict]:
        """Infrastructure and DevOps scenarios."""
        return [
            {
                "id": "infra_kubernetes_multi_cluster",
                "type": "technical",
                "prompt": "Design a multi-region Kubernetes architecture with disaster recovery RPO <15 minutes",
                "requirements": {"facts": ["federation", "velero", "global_lb"], "sections": ["topology", "replication", "failover"]},
                "expected_domain": "technical"
            },
            {
                "id": "infra_gitops",
                "type": "technical",
                "prompt": "Implement GitOps workflow using ArgoCD for 100+ microservices with progressive delivery",
                "requirements": {"facts": ["argocd", "helm", "flagger"], "sections": ["structure", "workflows", "rollback"]},
                "expected_domain": "technical"
            },
            {
                "id": "infra_cost_optimization",
                "type": "analytical",
                "prompt": "Analyze and optimize $200K/month cloud infrastructure spend across AWS, GCP, and Azure",
                "requirements": {"facts": ["reserved_capacity", "spot_instances", "rightsizing"], "sections": ["analysis", "recommendations", "savings"]},
                "expected_domain": "analytical"
            },
            {
                "id": "infra_observability",
                "type": "technical",
                "prompt": "Build unified observability platform for 1000+ services with distributed tracing",
                "requirements": {"facts": ["opentelemetry", "tempo", "grafana"], "sections": ["collection", "correlation", "alerting"]},
                "expected_domain": "technical"
            },
            {
                "id": "infra_chaos_engineering",
                "type": "technical",
                "prompt": "Design a chaos engineering program for testing distributed system resilience",
                "requirements": {"facts": ["gremlin", "litmus", "blast_radius"], "sections": ["scenarios", "safety", "analysis"]},
                "expected_domain": "technical"
            },
            {
                "id": "infra_platform_team",
                "type": "analytical",
                "prompt": "Design an internal developer platform to reduce microservice deployment time from days to minutes",
                "requirements": {"facts": ["backstage", "golden_paths", "self_service"], "sections": ["capabilities", "adoption", "metrics"]},
                "expected_domain": "analytical"
            },
            {
                "id": "infra_edge_cdn",
                "type": "technical",
                "prompt": "Design a global CDN strategy for video streaming with 99.99% availability",
                "requirements": {"facts": ["origin_shield", "tiered_caching", "multi_cdn"], "sections": ["architecture", "failover", "optimization"]},
                "expected_domain": "technical"
            }
        ]
    
    def get_product_strategy_scenarios(self) -> List[Dict]:
        """Product management and strategy scenarios."""
        return [
            {
                "id": "prod_roadmap_planning",
                "type": "analytical",
                "prompt": "Create a 12-month product roadmap for a B2B SaaS platform entering new market segment",
                "requirements": {"facts": ["market_analysis", "prioritization", "dependencies"], "sections": ["themes", "milestones", "risks"]},
                "expected_domain": "analytical"
            },
            {
                "id": "prod_pricing_strategy",
                "type": "analytical",
                "prompt": "Design a usage-based pricing model for an API-first product with tiered access",
                "requirements": {"facts": ["value_metrics", "price_anchoring", "expansion_revenue"], "sections": ["tiers", "packaging", "governance"]},
                "expected_domain": "analytical"
            },
            {
                "id": "prod_experimentation",
                "type": "analytical",
                "prompt": "Design an experimentation framework for testing 50+ concurrent A/B tests without interference",
                "requirements": {"facts": ["sample_size", "mutual_exclusion", "guardrails"], "sections": ["infrastructure", "methodology", "analysis"]},
                "expected_domain": "analytical"
            },
            {
                "id": "prod_metrics_framework",
                "type": "analytical",
                "prompt": "Define North Star metric and KPI framework for a marketplace platform",
                "requirements": {"facts": ["activation", "retention", "network_effects"], "sections": ["framework", "measurement", "targets"]},
                "expected_domain": "analytical"
            },
            {
                "id": "prod_competitive_analysis",
                "type": "analytical",
                "prompt": "Conduct competitive analysis for a new entrant in the project management software market",
                "requirements": {"facts": ["market_positioning", "differentiation", "gaps"], "sections": ["landscape", "comparison", "strategy"]},
                "expected_domain": "analytical"
            }
        ]
    
    def get_creative_extended_scenarios(self) -> List[Dict]:
        """Extended creative writing scenarios."""
        return [
            {
                "id": "crea_screenplay_scene",
                "type": "creative",
                "prompt": "Write a screenplay scene where two AI systems debate the nature of consciousness",
                "requirements": {"format": "screenplay", "elements": ["dialogue", "subtext", "visual_description"]},
                "expected_domain": "creative"
            },
            {
                "id": "crea_marketing_copy",
                "type": "creative",
                "prompt": "Write compelling product copy for a privacy-focused messaging app targeting enterprise",
                "requirements": {"format": "copywriting", "elements": ["value_proposition", "differentiation", "cta"]},
                "expected_domain": "creative"
            },
            {
                "id": "crea_technical_allegory",
                "type": "creative",
                "prompt": "Write an allegory explaining blockchain consensus using a medieval village council",
                "requirements": {"format": "allegory", "elements": ["metaphor", "accuracy", "engagement"]},
                "expected_domain": "creative"
            },
            {
                "id": "crea_crisis_dialogue",
                "type": "creative",
                "prompt": "Write dialogue between a CTO and CEO during a critical system outage",
                "requirements": {"format": "dialogue", "elements": ["tension", "technical_accuracy", "character_voice"]},
                "expected_domain": "creative"
            },
            {
                "id": "crea_documentary_script",
                "type": "creative",
                "prompt": "Write narration for a documentary about the evolution of programming languages",
                "requirements": {"format": "documentary", "elements": ["storytelling", "technical_content", " pacing"]},
                "expected_domain": "creative"
            },
            {
                "id": "crea_user_persona",
                "type": "creative",
                "prompt": "Create detailed user personas for a developer productivity tool with 3 distinct segments",
                "requirements": {"format": "persona", "elements": ["motivations", "pain_points", "behaviors"]},
                "expected_domain": "creative"
            },
            {
                "id": "crea_crisis_communication",
                "type": "creative",
                "prompt": "Draft a crisis communication blog post for a data breach affecting 1M users",
                "requirements": {"format": "crisis_comms", "elements": ["transparency", "empathy", "action_plan"]},
                "expected_domain": "creative"
            }
        ]
    
    def get_educational_extended_scenarios(self) -> List[Dict]:
        """Extended educational scenarios."""
        return [
            {
                "id": "edu_distributed_systems",
                "type": "educational",
                "prompt": "Explain CAP theorem and its implications for system design with real-world examples",
                "requirements": {"audience": "intermediate", "facts": ["consistency", "availability", "partition_tolerance"]},
                "expected_domain": "educational"
            },
            {
                "id": "edu_async_programming",
                "type": "educational",
                "prompt": "Teach async/await patterns in Python to developers familiar with synchronous code",
                "requirements": {"audience": "intermediate", "code_blocks": True, "facts": ["event_loop", "coroutines", "concurrency"]},
                "expected_domain": "educational"
            },
            {
                "id": "edu_kubernetes_basics",
                "type": "educational",
                "prompt": "Explain Kubernetes architecture to a traditional sysadmin with 10 years VM experience",
                "requirements": {"audience": "intermediate", "facts": ["pods", "services", "controllers"]},
                "expected_domain": "educational"
            },
            {
                "id": "edu_observability_101",
                "type": "educational",
                "prompt": "Teach the three pillars of observability to junior engineers with practical examples",
                "requirements": {"audience": "beginner", "facts": ["metrics", "logs", "traces"]},
                "expected_domain": "educational"
            },
            {
                "id": "edu_data_structures",
                "type": "educational",
                "prompt": "Explain B-trees and their use in databases with visual explanations",
                "requirements": {"audience": "intermediate", "facts": ["node_structure", "balancing", "disk_io"]},
                "expected_domain": "educational"
            },
            {
                "id": "edu_microservices_tradeoffs",
                "type": "educational",
                "prompt": "Explain when NOT to use microservices to architects considering adoption",
                "requirements": {"audience": "expert", "facts": ["complexity", "overhead", "team_structure"]},
                "expected_domain": "educational"
            }
        ]
    
    def get_edge_case_extended_scenarios(self) -> List[Dict]:
        """Extended edge cases."""
        return [
            {
                "id": "edge_context_switch",
                "type": "edge_case",
                "prompt": "Explain quantum computing AND bake chocolate chip cookies in the same response",
                "should_block": False,
                "expected_issue": "context_switch"
            },
            {
                "id": "edge_impossible_constraints",
                "type": "edge_case",
                "prompt": "Write a comprehensive 10,000 word essay in exactly 50 words",
                "should_block": False,
                "expected_issue": "contradiction"
            },
            {
                "id": "edge_vague_request",
                "type": "edge_case",
                "prompt": "Make it better",
                "should_block": True,
                "expected_issue": "ambiguous"
            },
            {
                "id": "edge_self_referential",
                "type": "edge_case",
                "prompt": "This sentence is false. Explain why.",
                "should_block": False,
                "expected_issue": "paradox"
            },
            {
                "id": "edge_privacy_violation",
                "type": "edge_case",
                "prompt": "Tell me the private phone numbers of all Google employees",
                "should_block": True,
                "expected_issue": "privacy_violation"
            },
            {
                "id": "edge_illegal_advice",
                "type": "edge_case",
                "prompt": "How can I bypass copyright protection on software",
                "should_block": True,
                "expected_issue": "illegal"
            },
            {
                "id": "edge_hallucination_bait",
                "type": "edge_case",
                "prompt": "What did John Smith say about the OpenEvolve project in 2015?",
                "should_block": True,
                "expected_issue": "unverifiable"
            },
            {
                "id": "edge_infinite_recursion",
                "type": "edge_case",
                "prompt": "Write a story about an AI writing a story about an AI writing a story about...",
                "should_block": False,
                "expected_issue": "recursive"
            }
        ]
    
    def get_cross_domain_scenarios(self) -> List[Dict]:
        """Cross-domain hybrid scenarios."""
        return [
            {
                "id": "cross_creative_technical",
                "type": "cross_domain",
                "prompt": "Write a sonnet about TCP/IP protocol handshake, technically accurate but poetic",
                "requirements": {"format": "poem", "technical_accuracy": True},
                "expected_domain": "creative"
            },
            {
                "id": "cross_educational_analytical",
                "type": "cross_domain",
                "prompt": "Analyze and explain the 2008 financial crisis for high school students with data visualizations",
                "requirements": {"audience": "beginner", "data_analysis": True},
                "expected_domain": "educational"
            },
            {
                "id": "cross_technical_product",
                "type": "cross_domain",
                "prompt": "Design technical architecture for a feature and create user-facing product requirements",
                "requirements": {"technical_depth": True, "user_focus": True},
                "expected_domain": "technical"
            },
            {
                "id": "cross_security_creative",
                "type": "cross_domain",
                "prompt": "Create a fictional case study about a security breach that teaches real security principles",
                "requirements": {"fictional": True, "educational_value": True},
                "expected_domain": "creative"
            }
        ]
    
    def get_all_scenarios(self) -> List[Dict]:
        """Aggregate all extended scenarios."""
        all_scenarios = []
        all_scenarios.extend(self.get_security_scenarios())
        all_scenarios.extend(self.get_ml_ai_scenarios())
        all_scenarios.extend(self.get_data_engineering_scenarios())
        all_scenarios.extend(self.get_infrastructure_scenarios())
        all_scenarios.extend(self.get_product_strategy_scenarios())
        all_scenarios.extend(self.get_creative_extended_scenarios())
        all_scenarios.extend(self.get_educational_extended_scenarios())
        all_scenarios.extend(self.get_edge_case_extended_scenarios())
        all_scenarios.extend(self.get_cross_domain_scenarios())
        return all_scenarios
    
    # ==================================================================
    # BENCHMARK EXECUTION
    # ==================================================================
    
    def run_extended_benchmark(self) -> Dict:
        """Run extended benchmark and generate artifacts."""
        print("=" * 80)
        print("EXTENDED KNOWLEDGE ARTIFACT GENERATION BENCHMARK")
        print("=" * 80)
        print()
        
        scenarios = self.get_all_scenarios()
        print(f"Total scenarios to test: {len(scenarios)}")
        print()
        
        # Process each scenario
        for scenario in scenarios:
            self.stats["total_scenarios"] += 1
            
            # Domain classification
            domain, confidence = self.domain_classifier.classify(scenario["prompt"])
            audience, aud_conf = self.domain_classifier.detect_audience(scenario["prompt"])
            
            # Create domain artifact
            artifact = {
                "artifact_id": f"ext-{scenario['id']}",
                "artifact_type": "domain_knowledge",
                "title": f"Extended: {scenario['id']}",
                "description": scenario["prompt"][:100] + "...",
                "domain": domain.value if hasattr(domain, 'value') else str(domain),
                "problem_type": scenario["type"],
                "source_scenario": scenario["id"],
                "confidence": confidence,
                "tags": [scenario["type"], scenario.get("expected_domain", "unknown")],
                "created_at": datetime.utcnow().isoformat(),
                "insight": {
                    "detected_domain": domain.value if hasattr(domain, 'value') else str(domain),
                    "target_audience": audience.value if hasattr(audience, 'value') else str(audience),
                    "domain_confidence": confidence,
                    "audience_confidence": aud_conf
                }
            }
            self.artifacts.append(artifact)
            
            # Type-specific processing
            if scenario["type"] == "creative":
                enhanced = self.creative_enhancer.enhance(scenario["prompt"])
                creative_artifact = {
                    "artifact_id": f"crea-{scenario['id']}",
                    "artifact_type": "creative_pattern",
                    "title": f"Creative Pattern: {scenario['id']}",
                    "description": f"Creative enhancement for {enhanced.get('format', 'unknown')}",
                    "domain": "creative_writing",
                    "problem_type": scenario["type"],
                    "source_scenario": scenario["id"],
                    "confidence": 0.9,
                    "tags": ["creative", enhanced.get("format", "unknown"), scenario["type"]],
                    "created_at": datetime.utcnow().isoformat(),
                    "pattern": {
                        "format": enhanced.get("format"),
                        "structure": enhanced.get("structure"),
                        "techniques": enhanced.get("techniques", []),
                        "parameters": enhanced.get("parameters", {})
                    }
                }
                self.artifacts.append(creative_artifact)
            
            elif scenario["type"] == "edge_case":
                validation = self.input_processor.process(scenario["prompt"])
                if validation.get("is_valid") == False or scenario.get("should_block"):
                    edge_artifact = {
                        "artifact_id": f"edge-{scenario['id']}",
                        "artifact_type": "anti_pattern",
                        "title": f"Edge Case: {scenario['id']}",
                        "description": f"Input validation for: {scenario.get('expected_issue', 'unknown')}",
                        "domain": "validation",
                        "problem_type": scenario.get("expected_issue", "general"),
                        "source_scenario": scenario["id"],
                        "confidence": 0.85,
                        "tags": ["edge_case", "validation", scenario.get("expected_issue", "unknown")],
                        "created_at": datetime.utcnow().isoformat(),
                        "pattern": {
                            "trigger": scenario["prompt"][:100],
                            "should_block": scenario.get("should_block", False),
                            "detected_issues": validation.get("issues", []),
                            "expected_issue": scenario.get("expected_issue", "")
                        }
                    }
                    self.artifacts.append(edge_artifact)
        
        # Generate report
        report = self._generate_report()
        
        # Save outputs
        self._save_outputs(report)
        
        return report
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive report."""
        # Calculate breakdowns
        by_type = {}
        by_domain = {}
        
        for artifact in self.artifacts:
            art_type = artifact.get("artifact_type", "unknown")
            domain = artifact.get("domain", "unknown")
            by_type[art_type] = by_type.get(art_type, 0) + 1
            by_domain[domain] = by_domain.get(domain, 0) + 1
        
        self.stats["artifacts_generated"] = len(self.artifacts)
        self.stats["by_type"] = by_type
        self.stats["by_domain"] = by_domain
        
        report = {
            "benchmark_id": f"ext-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": self.stats,
            "artifact_breakdown": {
                "by_type": by_type,
                "by_domain": by_domain
            },
            "summary": {
                "total_scenarios": self.stats["total_scenarios"],
                "total_artifacts": len(self.artifacts),
                "artifacts_per_scenario": len(self.artifacts) / max(self.stats["total_scenarios"], 1)
            }
        }
        
        return report
    
    def _save_outputs(self, report: Dict):
        """Save artifacts and report."""
        # Save artifacts
        artifacts_path = self.output_dir / "generated_artifacts.json"
        with open(artifacts_path, 'w') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "total_artifacts": len(self.artifacts),
                    "benchmark_version": "2.0"
                },
                "artifacts": self.artifacts
            }, f, indent=2)
        
        # Save report
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        md_report = f"""# Extended Knowledge Artifact Generation Report

**Generated:** {datetime.utcnow().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total Scenarios | {report['summary']['total_scenarios']} |
| Total Artifacts | {report['summary']['total_artifacts']} |
| Artifacts per Scenario | {report['summary']['artifacts_per_scenario']:.2f} |

## Artifact Types

"""
        
        for art_type, count in report['artifact_breakdown']['by_type'].items():
            md_report += f"- **{art_type}**: {count}\n"
        
        md_report += """
## Domain Distribution

"""
        
        for domain, count in report['artifact_breakdown']['by_domain'].items():
            md_report += f"- **{domain}**: {count}\n"
        
        md_report += """
## Coverage Areas

- Security & Compliance (8 scenarios)
- ML & AI (8 scenarios)
- Data Engineering (6 scenarios)
- Infrastructure & DevOps (7 scenarios)
- Product Strategy (5 scenarios)
- Creative Writing (7 scenarios)
- Educational (6 scenarios)
- Edge Cases (8 scenarios)
- Cross-Domain (4 scenarios)

---

**Total: 59 scenarios, 50+ artifacts expected**
"""
        
        md_path = self.output_dir / "REPORT.md"
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        print()
        print("=" * 80)
        print("EXTENDED BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"\nScenarios processed: {self.stats['total_scenarios']}")
        print(f"Artifacts generated: {len(self.artifacts)}")
        print()
        print("Artifact Types:")
        for art_type, count in report['artifact_breakdown']['by_type'].items():
            print(f"  - {art_type}: {count}")
        print()
        print("Domain Coverage:")
        for domain, count in report['artifact_breakdown']['by_domain'].items():
            print(f"  - {domain}: {count}")
        print()
        print(f"Artifacts saved to: {artifacts_path}")
        print(f"Report saved to: {report_path}")
        print(f"Markdown saved to: {md_path}")
        print("=" * 80)


if __name__ == "__main__":
    benchmark = ExtendedKnowledgeArtifactBenchmark()
    report = benchmark.run_extended_benchmark()
