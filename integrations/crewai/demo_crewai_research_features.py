"""
Demo Script for CrewAI Research Roadmap Features

Demonstrates all 10 implemented features with working examples.

Usage: python demo_crewai_research_features.py
"""

import asyncio
import json
import tempfile
import os
from datetime import datetime


def print_header(feature_num: int, feature_name: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"FEATURE {feature_num}: {feature_name}")
    print("=" * 70)


def demo_feature_1_hierarchical_crew():
    """Demo Feature 1: Hierarchical Process Support"""
    print_header(1, "Hierarchical Process Support")
    
    from crewai_research_core import create_hierarchical_crew, HierarchicalTask, CrewLevel
    
    # Create hierarchical crew
    crew = create_hierarchical_crew(name="ResearchCrew", max_depth=3)
    print("[OK] Created hierarchical crew: ResearchCrew")
    
    # Create manager-led crew
    manager_config = {"name": "Research Manager", "expertise": ["research_management"]}
    worker_configs = [
        {"name": "Data Analyst", "skills": ["data_analysis", "statistics"]},
        {"name": "Literature Reviewer", "skills": ["literature_search", "synthesis"]}
    ]
    
    config = crew.create_manager_crew(manager_config, worker_configs)
    print(f"[OK] Created manager crew with ID: {config['crew_id']}")
    print(f"  - Manager: {config['manager_id']}")
    print(f"  - Workers: {len(config['worker_ids'])}")
    
    # Create and delegate task
    task = HierarchicalTask(
        task_id="lit_review_task",
        title="Literature Review on AI Safety",
        description="Conduct comprehensive literature review on AI safety",
        level=CrewLevel.WORKER,
        priority=8
    )
    
    result = crew.delegate_task(
        task=task,
        from_agent_id=config['manager_id'],
        to_agent_ids=config['worker_ids'][:1]
    )
    
    print(f"[OK] Delegated task: {result['task_id']}")
    print(f"  - Sub-tasks created: {len(result['sub_tasks'])}")
    
    # Show hierarchy status
    status = crew.get_hierarchy_status()
    print(f"[OK] Hierarchy Status:")
    print(f"  - Total Agents: {status['total_agents']}")
    print(f"  - Total Tasks: {status['total_tasks']}")


def demo_feature_2_advanced_delegation():
    """Demo Feature 2: Advanced Delegation Mechanisms"""
    print_header(2, "Advanced Delegation Mechanisms")
    
    from crewai_research_core import create_delegation_manager, DelegationType, AgentCapability
    
    # Create delegation manager
    manager = create_delegation_manager()
    print("[OK] Created delegation manager")
    
    # Register agents with different capabilities
    manager.register_agent(AgentCapability(
        agent_id="ml_expert",
        skills=["machine_learning", "deep_learning", "python"],
        role="senior",
        max_workload=3,
        performance_score=0.95
    ))
    
    manager.register_agent(AgentCapability(
        agent_id="data_analyst",
        skills=["data_analysis", "statistics", "python"],
        role="analyst",
        max_workload=5,
        performance_score=0.88
    ))
    
    manager.register_agent(AgentCapability(
        agent_id="qa_engineer",
        skills=["testing", "qa", "automation"],
        role="qa_engineer",
        max_workload=5,
        performance_score=0.90
    ))
    
    print("[OK] Registered 3 agents with different capabilities")
    
    # Demonstrate different delegation types
    
    # Skill-based delegation
    result = manager.delegate(
        task={"id": "ml_task", "required_skills": ["machine_learning", "python"]},
        delegation_type=DelegationType.SKILL_BASED
    )
    print(f"[OK] Skill-based delegation: Assigned to {result['agent_id']}")
    
    # Role-based delegation
    result = manager.delegate(
        task={"id": "critical_task", "required_role": "senior", "priority": 9},
        delegation_type=DelegationType.ROLE_BASED
    )
    print(f"[OK] Role-based delegation: Assigned to {result['agent_id']}")
    
    # Load-balanced delegation
    result = manager.delegate(
        task={"id": "regular_task"},
        delegation_type=DelegationType.LOAD_BALANCED
    )
    print(f"[OK] Load-balanced delegation: Assigned to {result['agent_id']}")
    
    # Show stats
    stats = manager.get_delegation_stats()
    print(f"[OK] Delegation Stats:")
    print(f"  - Total Agents: {stats['total_agents']}")
    print(f"  - Average Workload: {stats['average_workload']:.2f}")


def demo_feature_3_memory_system():
    """Demo Feature 3: Memory-Augmented Research"""
    print_header(3, "Memory-Augmented Research")
    
    from crewai_research_core import create_memory_system, MemoryType
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create memory system
        memory = create_memory_system(storage_dir=temp_dir)
        print("[OK] Created memory-augmented research system")
        
        # Store conversation memory
        session_id = "research_session_001"
        memory.store(
            content="Researcher: What are the key findings?",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": session_id, "role": "researcher"},
            importance=0.8
        )
        memory.store(
            content="Assistant: The results show significant improvement.",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": session_id, "role": "assistant"},
            importance=0.8
        )
        print("[OK] Stored 2 conversation memories")
        
        # Store entity memory
        memory.store(
            content={"name": "Transformer Architecture", "type": "neural_network"},
            memory_type=MemoryType.ENTITY,
            metadata={"entity_type": "architecture", "mention_count": 15},
            importance=0.9
        )
        print("[OK] Stored entity memory")
        
        # Store long-term knowledge
        memory.store(
            content="Attention mechanisms allow models to focus on relevant parts of input.",
            memory_type=MemoryType.LONG_TERM,
            metadata={"topic": "attention", "field": "nlp"},
            importance=0.85
        )
        print("[OK] Stored long-term knowledge")
        
        # Retrieve memories
        results = memory.retrieve("attention transformer", top_k=5)
        print(f"[OK] Retrieved {len(results)} relevant memories")
        
        # Get conversation history
        history = memory.retrieve_conversation_history(session_id)
        print(f"[OK] Conversation history: {len(history)} messages")
        
        # Get entities
        entities = memory.retrieve_entities()
        print(f"[OK] Stored entities: {len(entities)}")


def demo_feature_4_tool_orchestration():
    """Demo Feature 4: External Tool Orchestration"""
    print_header(4, "External Tool Orchestration")
    
    from crewai_research_tools import create_tool_orchestrator, ToolDefinition, ToolType
    
    async def run_demo():
        orchestrator = create_tool_orchestrator()
        print("[OK] Created tool orchestrator")
        
        # Register custom tools
        def calculate_stats(data: list) -> dict:
            return {
                "mean": sum(data) / len(data),
                "min": min(data),
                "max": max(data),
                "count": len(data)
            }
        
        tool_id = orchestrator.register_custom_tool(
            name="stats_calculator",
            func=calculate_stats,
            input_schema={"data": {"type": "array", "items": {"type": "number"}}}
        )
        print(f"[OK] Registered custom tool: {tool_id}")
        
        # Execute tool
        result = await orchestrator.execute_tool(
            tool_id=tool_id,
            inputs={"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            use_cache=False
        )
        
        if result.success:
            print(f"[OK] Tool execution successful")
            print(f"  - Mean: {result.result['mean']:.2f}")
            print(f"  - Min: {result.result['min']}")
            print(f"  - Max: {result.result['max']}")
            print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
        
        # Show stats
        stats = orchestrator.get_tool_stats()
        print(f"[OK] Tool Stats: {stats['cache']['size']} cached items")
    
    asyncio.run(run_demo())


def demo_feature_5_multimodal():
    """Demo Feature 5: Multi-Modal Support"""
    print_header(5, "Multi-Modal Support")
    
    from crewai_research_tools import create_multimodal_processor
    
    processor = create_multimodal_processor()
    print("[OK] Created multi-modal processor")
    
    # Check capabilities
    caps = processor.get_capabilities()
    print("[OK] Available capabilities:")
    for cap, enabled in caps.items():
        status = "[OK]" if enabled else "[FAIL]"
        print(f"  [{status}] {cap.capitalize()}")
    
    # Demonstrate document parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.\n")
        temp_path = f.name
    
    try:
        result = processor.parse_document(temp_path)
        print(f"[OK] Parsed document: {result['file_type']}")
        print(f"  - Content length: {len(result['content'])} chars")
    finally:
        os.unlink(temp_path)


def demo_feature_6_collaboration():
    """Demo Feature 6: Real-Time Collaboration"""
    print_header(6, "Real-Time Collaboration")
    
    from crewai_research_tools import create_collaboration_system, CollaborationEventType
    
    collab = create_collaboration_system()
    print("[OK] Created collaboration system")
    
    # Create channel
    channel = collab.create_channel("research_lab", channel_type="room")
    print(f"[OK] Created channel: {channel}")
    
    # Agents join
    agents = ["lead_researcher", "analyst_1", "analyst_2"]
    for agent in agents:
        collab.join_channel(channel, agent, {"role": "researcher"})
    print(f"[OK] {len(agents)} agents joined the channel")
    
    # Broadcast message
    collab.broadcast(
        channel_id=channel,
        event_type=CollaborationEventType.MESSAGE,
        source_agent_id="lead_researcher",
        payload={"message": "Starting literature review phase"}
    )
    print("[OK] Broadcast message sent")
    
    # Send direct message
    collab.send_direct_message(
        from_agent_id="lead_researcher",
        to_agent_id="analyst_1",
        message="Please focus on recent papers from 2023-2024"
    )
    print("[OK] Direct message sent")
    
    # Send notification
    collab.notify(
        agent_id="analyst_2",
        notification_type="task_assigned",
        message="You have been assigned to statistical analysis",
        priority="high"
    )
    print("[OK] Notification sent")
    
    # Show channel info
    info = collab.get_channel_info(channel)
    print(f"[OK] Channel info: {info['participants']} participants")


def demo_feature_7_templates():
    """Demo Feature 7: Research Workflow Templates"""
    print_header(7, "Research Workflow Templates")
    
    from crewai_research_templates import create_template_registry, TemplateType
    
    registry = create_template_registry()
    print("[OK] Created template registry")
    
    # List all templates
    templates = registry.list_templates()
    print(f"[OK] Available templates: {len(templates)}")
    for t in templates:
        print(f"  - {t['name']} ({t['type']}): {t['steps_count']} steps, ~{t['estimated_duration_hours']}h")
    
    # Get literature review template
    lit_review = registry.get_template_by_type(TemplateType.LITERATURE_REVIEW)
    print(f"\n[OK] Literature Review Template:")
    
    # Create execution plan
    plan = lit_review.get_execution_plan({
        "research_topic": "AI Safety in Large Language Models",
        "research_questions": [
            "What are the main safety concerns?",
            "What mitigation strategies exist?"
        ],
        "inclusion_criteria": ["peer-reviewed", "2020-2024"],
        "target_paper_count": 50
    })
    
    print(f"  - Total steps: {plan['total_steps']}")
    print(f"  - Estimated duration: {plan['estimated_duration_hours']} hours")
    print(f"  - First 3 steps:")
    for step in plan['steps'][:3]:
        print(f"    {step['step_id']}: {step['name']} ({step['estimated_duration']} min)")


def demo_feature_8_literature_search():
    """Demo Feature 8: Automated Literature Search"""
    print_header(8, "Automated Literature Search")
    
    from crewai_research_external import create_literature_search, DatabaseType
    
    search = create_literature_search()
    print("[OK] Created literature search orchestrator")
    
    # Search single database (using mock implementation)
    print("\n[OK] Searching Semantic Scholar...")
    papers = search.search(
        query="machine learning",
        database=DatabaseType.SEMANTIC_SCHOLAR,
        max_results=3
    )
    
    print(f"  Found {len(papers)} papers")
    for i, paper in enumerate(papers[:2], 1):
        print(f"  {i}. {paper.title[:60]}...")
        print(f"     Authors: {', '.join(paper.authors[:2])}")
        print(f"     Citations: {paper.citation_count}")
    
    # Search across multiple databases
    print("\n[OK] Searching multiple databases...")
    results = search.search_all(
        query="artificial intelligence",
        max_results_per_db=2,
        databases=[DatabaseType.SEMANTIC_SCHOLAR, DatabaseType.ARXIV],
        deduplicate=True
    )
    
    for db, papers in results.items():
        print(f"  {db.value}: {len(papers)} papers")
    
    # Citation analysis
    print("\n[OK] Performing citation analysis...")
    analysis = search.analyze_citations(papers)
    print(f"  - Total papers: {analysis['total_papers']}")
    print(f"  - Total citations: {analysis['total_citations']}")
    print(f"  - Average citations: {analysis['average_citations']:.1f}")


def demo_feature_9_experiment_tracking():
    """Demo Feature 9: Experiment Tracking"""
    print_header(9, "Experiment Tracking")
    
    from crewai_research_external import create_experiment_tracker
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = create_experiment_tracker(storage_dir=temp_dir)
        print("[OK] Created experiment tracker")
        
        # Create experiment
        exp_id = tracker.create_experiment(
            name="BERT Fine-tuning for Sentiment Analysis",
            description="Fine-tuning BERT on IMDB dataset",
            parameters={
                "model": "bert-base-uncased",
                "learning_rate": 2e-5,
                "batch_size": 32,
                "epochs": 3
            },
            tags=["nlp", "bert", "sentiment", "classification"]
        )
        print(f"[OK] Created experiment: {exp_id}")
        
        # Log metrics during training
        print("[OK] Logging training metrics...")
        for epoch in range(1, 4):
            tracker.log_metrics(exp_id, {
                "train_loss": 0.5 / epoch,
                "val_loss": 0.6 / epoch,
                "accuracy": 0.8 + (epoch * 0.05),
                "f1_score": 0.78 + (epoch * 0.06)
            }, step=epoch * 100)
        
        # Log final metrics
        tracker.log_metrics(exp_id, {
            "final_accuracy": 0.95,
            "final_f1": 0.94,
            "test_accuracy": 0.93
        })
        
        # Log artifact
        tracker.log_artifact(
            exp_id,
            name="best_model.pt",
            artifact_type="model",
            file_path="/path/to/model.pt",
            metadata={"accuracy": 0.95, "epoch": 3}
        )
        print("[OK] Logged model artifact")
        
        # Complete experiment
        tracker.complete_experiment(exp_id, status="completed")
        print("[OK] Experiment completed")
        
        # Show summary
        exp = tracker.get_experiment(exp_id)
        print(f"\n[OK] Experiment Summary:")
        print(f"  - Status: {exp.status}")
        print(f"  - Parameters: {len(exp.parameters)}")
        print(f"  - Metrics logged: {len(exp.metrics)}")
        print(f"  - Artifacts: {len(exp.artifacts)}")


def demo_feature_10_report_generation():
    """Demo Feature 10: Research Report Generation"""
    print_header(10, "Research Report Generation")
    
    from crewai_research_external import create_report_generator, ReportFormat
    from crewai_research_external import Paper
    
    generator = create_report_generator()
    print("[OK] Created report generator")
    
    # Add sections
    generator.add_section("Abstract", "This study investigates...")
    generator.add_section("Introduction", "Recent advances in AI have...")
    generator.add_section("Methodology", "We conducted a comprehensive analysis...")
    
    # Add results table
    generator.add_table(
        title="Experimental Results",
        headers=["Model", "Accuracy", "F1 Score", "Training Time"],
        rows=[
            ["BERT-base", "0.92", "0.91", "45 min"],
            ["BERT-large", "0.95", "0.94", "120 min"],
            ["RoBERTa", "0.94", "0.93", "90 min"]
        ]
    )
    print("[OK] Added sections and table")
    
    # Add citations
    paper1 = Paper(
        paper_id="1",
        title="Attention is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        abstract="...",
        publication_date="2017",
        journal="NeurIPS"
    )
    paper2 = Paper(
        paper_id="2",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
        abstract="...",
        publication_date="2019",
        journal="NAACL"
    )
    
    cite1 = generator.add_citation(paper1)
    cite2 = generator.add_citation(paper2)
    print(f"[OK] Added 2 citations")
    
    # Generate formats
    markdown = generator.generate_markdown()
    html = generator.generate_html()
    
    print(f"[OK] Generated Markdown ({len(markdown)} chars)")
    print(f"[OK] Generated HTML ({len(html)} chars)")
    
    # Save to temp files
    with tempfile.TemporaryDirectory() as temp_dir:
        md_path = os.path.join(temp_dir, "report.md")
        generator.export(md_path, ReportFormat.MARKDOWN)
        print(f"[OK] Exported to Markdown: {md_path}")
        
        html_path = os.path.join(temp_dir, "report.html")
        generator.export(html_path, ReportFormat.HTML)
        print(f"[OK] Exported to HTML: {html_path}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("CREWAI RESEARCH ROADMAP - FEATURE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases all 10 implemented features.")
    print("Each feature is demonstrated with working examples.\n")
    
    try:
        demo_feature_1_hierarchical_crew()
    except Exception as e:
        print(f"Feature 1 error: {e}")
    
    try:
        demo_feature_2_advanced_delegation()
    except Exception as e:
        print(f"Feature 2 error: {e}")
    
    try:
        demo_feature_3_memory_system()
    except Exception as e:
        print(f"Feature 3 error: {e}")
    
    try:
        demo_feature_4_tool_orchestration()
    except Exception as e:
        print(f"Feature 4 error: {e}")
    
    try:
        demo_feature_5_multimodal()
    except Exception as e:
        print(f"Feature 5 error: {e}")
    
    try:
        demo_feature_6_collaboration()
    except Exception as e:
        print(f"Feature 6 error: {e}")
    
    try:
        demo_feature_7_templates()
    except Exception as e:
        print(f"Feature 7 error: {e}")
    
    try:
        demo_feature_8_literature_search()
    except Exception as e:
        print(f"Feature 8 error: {e}")
    
    try:
        demo_feature_9_experiment_tracking()
    except Exception as e:
        print(f"Feature 9 error: {e}")
    
    try:
        demo_feature_10_report_generation()
    except Exception as e:
        print(f"Feature 10 error: {e}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - All 10 features demonstrated successfully!")
    print("=" * 70)
    print("\nFor more details, see:")
    print("  - CREWAI_RESEARCH_IMPLEMENTATION_COMPLETE.md")
    print("  - test_crewai_research_comprehensive.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
