"""
Migration Script for Z3 Knowledge Integration

Creates database tables, indexes, and initial data for Z3 knowledge storage.

Usage:
    python -m knowledge_engine.integrations.z3_migration --create
    python -m knowledge_engine.integrations.z3_migration --drop
    python -m knowledge_engine.integrations.z3_migration --seed

Author: OpenEvolve
Created: 2026-01-31
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration(
    action: str = "create",
    database_url: str = "sqlite:///z3_knowledge.db",
    seed_data: bool = False
):
    """
    Run database migration.
    
    Args:
        action: 'create', 'drop', or 'recreate'
        database_url: Database connection URL
        seed_data: Whether to seed with initial data
    """
    try:
        from knowledge_engine.integrations.z3_database_models import (
            create_z3_tables,
            drop_z3_tables,
            Z3KnowledgeEntry,
            Z3ProofPattern,
            Z3ConstraintPattern,
            Z3Strategy,
            Z3MathematicalInsight,
            Z3SolverResult
        )
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    
    try:
        if action in ("drop", "recreate"):
            logger.info(f"Dropping Z3 knowledge tables from {database_url}")
            drop_z3_tables(database_url)
        
        if action in ("create", "recreate"):
            logger.info(f"Creating Z3 knowledge tables in {database_url}")
            engine = create_z3_tables(database_url)
            
            if seed_data:
                logger.info("Seeding with initial data")
                _seed_initial_data(engine)
        
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def _seed_initial_data(engine):
    """Seed database with initial Z3 knowledge patterns."""
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Common proof patterns
    proof_patterns = [
        {
            "name": "Simplify-Solve-SMT",
            "description": "Standard simplification followed by equation solving and SMT",
            "tactics": ["simplify", "solve-eqs", "smt"],
            "domains": ["linear", "general"],
            "effectiveness": 0.92
        },
        {
            "name": "Bit-Vector Simplification",
            "description": "Pattern for bit-vector manipulation problems",
            "tactics": ["bv-simplify", "bit-blast", "smt"],
            "domains": ["bitvector", "hardware"],
            "effectiveness": 0.88
        },
        {
            "name": "Quantifier Elimination",
            "description": "Pattern for eliminating universal/existential quantifiers",
            "tactics": ["qe", "simplify", "smt"],
            "domains": ["theorem_proving", "logic"],
            "effectiveness": 0.85
        }
    ]
    
    for pattern_data in proof_patterns:
        entry = Z3KnowledgeEntry(
            entry_type="proof_pattern",
            content_hash=f"seed_proof_{pattern_data['name']}",
            content=pattern_data["description"],
            metadata_json={"seed": True, "version": "1.0"},
            problem_domain=",".join(pattern_data["domains"]),
            confidence=pattern_data["effectiveness"]
        )
        session.add(entry)
        session.flush()  # Get entry.id
        
        pattern = Z3ProofPattern(
            knowledge_entry_id=entry.id,
            pattern_signature=f"seed_{pattern_data['name']}",
            pattern_name=pattern_data["name"],
            description=pattern_data["description"],
            tactic_sequence=pattern_data["tactics"],
            applicable_domains=pattern_data["domains"],
            effectiveness_score=pattern_data["effectiveness"],
            usage_count=0
        )
        session.add(pattern)
    
    # Common constraint patterns
    constraint_patterns = [
        {
            "type": "linear",
            "structure": "Linear inequalities with integer coefficients",
            "complexity": 1.0,
            "typical_time": 0.1
        },
        {
            "type": "nonlinear",
            "structure": "Polynomial equations with degree >= 2",
            "complexity": 3.5,
            "typical_time": 2.5
        },
        {
            "type": "boolean",
            "structure": "Logical combinations of boolean variables",
            "complexity": 1.5,
            "typical_time": 0.5
        }
    ]
    
    for cp_data in constraint_patterns:
        entry = Z3KnowledgeEntry(
            entry_type="constraint",
            content_hash=f"seed_constraint_{cp_data['type']}",
            content=cp_data["structure"],
            metadata_json={"seed": True, "version": "1.0"},
            problem_domain=cp_data["type"],
            confidence=0.8
        )
        session.add(entry)
        session.flush()
        
        pattern = Z3ConstraintPattern(
            knowledge_entry_id=entry.id,
            pattern_type=cp_data["type"],
            structure_template=cp_data["structure"],
            complexity_score=cp_data["complexity"],
            typical_solving_time=cp_data["typical_time"],
            frequency=1
        )
        session.add(pattern)
    
    # Sample strategies
    strategies = [
        {
            "name": "Fast Linear Solver",
            "pattern": "linear_vars_<_10",
            "tactics": ["simplify", "solve-eqs", "smt"],
            "config": {"timeout": 10, "threads": 2},
            "expected_time": 0.5
        },
        {
            "name": "Complex Nonlinear Handler",
            "pattern": "nonlinear_vars_>=_5",
            "tactics": ["simplify", "nlsat", "smt"],
            "config": {"timeout": 60, "threads": 4},
            "expected_time": 5.0
        }
    ]
    
    for strat_data in strategies:
        entry = Z3KnowledgeEntry(
            entry_type="strategy",
            content_hash=f"seed_strategy_{strat_data['name']}",
            content=strat_data["name"],
            metadata_json={"seed": True, "version": "1.0"},
            problem_domain="general",
            confidence=0.75
        )
        session.add(entry)
        session.flush()
        
        strategy = Z3Strategy(
            knowledge_entry_id=entry.id,
            strategy_name=strat_data["name"],
            problem_pattern=strat_data["pattern"],
            recommended_tactics=strat_data["tactics"],
            solver_configuration=strat_data["config"],
            expected_avg_time=strat_data["expected_time"],
            success_count=5,
            failure_count=1
        )
        session.add(strategy)
    
    session.commit()
    session.close()
    
    logger.info("Seeded database with initial Z3 knowledge patterns")


def verify_migration(database_url: str = "sqlite:///z3_knowledge.db"):
    """Verify migration was successful."""
    try:
        from sqlalchemy import create_engine, inspect
        
        engine = create_engine(database_url)
        inspector = inspect(engine)
        
        expected_tables = [
            'z3_knowledge_entries',
            'z3_proof_patterns',
            'z3_constraint_patterns',
            'z3_strategies',
            'z3_mathematical_insights',
            'z3_solver_results',
            'z3_kg_nodes',
            'z3_kg_edges'
        ]
        
        existing_tables = inspector.get_table_names()
        
        print("\nMigration Verification")
        print("=" * 50)
        
        all_present = True
        for table in expected_tables:
            status = "✓" if table in existing_tables else "✗"
            print(f"  {status} {table}")
            if table not in existing_tables:
                all_present = False
        
        print("=" * 50)
        if all_present:
            print("All tables present - migration successful!")
            return True
        else:
            print("Some tables missing - migration incomplete")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Z3 Knowledge Database Migration"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create tables"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop tables"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate tables"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed with initial data"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration"
    )
    parser.add_argument(
        "--database",
        default="sqlite:///z3_knowledge.db",
        help="Database URL"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_migration(args.database)
        sys.exit(0 if success else 1)
    
    action = None
    if args.drop:
        action = "drop"
    elif args.recreate:
        action = "recreate"
    elif args.create:
        action = "create"
    
    if action:
        success = run_migration(
            action=action,
            database_url=args.database,
            seed_data=args.seed
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
