"""
Sovereign-Grade Problem Decomposition System - Web UI
Flask-based interface for monitoring and visualizing decomposition workflows.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from dependency_manager import DependencyManager
from sovereign_team_coordination import TeamCoordinator
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_gauntlets import GauntletSystem
from sovereign_data_models import (
    ProblemDefinition, 
    DecompositionPlan, 
    SubProblem,
    generate_id
)
from sovereign_persistence import SovereignDatabase
from sovereign_reliability import get_error_handler, get_health_monitor


# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sovereign-decomposition-key')
CORS(app)

# Initialize components
db = SovereignDatabase()
error_handler = get_error_handler()
health_monitor = get_health_monitor()
problem_analyzer = ProblemAnalyzer()
decomposition_engine = DecompositionEngine(problem_analyzer=problem_analyzer)
dependency_manager = DependencyManager()
team_coordinator = TeamCoordinator()
solution_orchestrator = SolutionOrchestrator()
gauntlet_system = GauntletSystem()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('sovereign_dashboard.html')


@app.route('/api/problems')
def get_problems():
    """Get list of all problems in the system"""
    try:
        problems = db.list_problems()
        return jsonify([p.to_dict() for p in problems])
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/problems', methods=['POST'])
def create_problem():
    """Create a new problem definition"""
    try:
        data = request.json
        problem_text = data.get('problem_text', '')
        title = data.get('title', '')
        
        # Analyze the problem
        problem = problem_analyzer.analyze_problem(problem_text, title)
        
        # Save to database
        db.create_problem(problem)
        
        return jsonify(problem.to_dict())
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/problems/<problem_id>/decompose', methods=['POST'])
def decompose_problem(problem_id):
    """Decompose a problem using the decomposition engine"""
    try:
        # Get the problem from database
        problem = db.get_problem(problem_id)
        if not problem:
            return jsonify({'error': f'Problem with ID {problem_id} not found'}), 404
        
        # Get strategy from request or use default
        strategy = request.json.get('strategy', 'hybrid')
        
        # Decompose the problem
        plan = decomposition_engine.decompose(problem, strategy)
        
        # Build dependency graph
        graph = dependency_manager.build_graph(plan.sub_problems)
        plan.dependency_graph = graph
        
        # Validate dependencies
        validation_result = dependency_manager.validate_dependencies(graph)
        if not validation_result.passed:
            # Handle cyclic dependencies
            cycles = dependency_manager.detect_cycles(graph)
            if cycles:
                # Try to resolve cycles
                for cycle in cycles:
                    # In a real implementation, we would have more sophisticated cycle resolution
                    app.logger.warning(f"Detected cycle: {cycle}")
        
        # Save plan to database
        db.create_plan(plan)
        
        return jsonify(plan.to_dict())
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/plans/<plan_id>')
def get_plan(plan_id):
    """Get a specific decomposition plan"""
    try:
        plan = db.get_plan(plan_id)
        if not plan:
            return jsonify({'error': f'Plan with ID {plan_id} not found'}), 404
        return jsonify(plan.to_dict())
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/plans/<plan_id>/execute', methods=['POST'])
def execute_plan(plan_id):
    """Execute a decomposition plan with team validation"""
    try:
        plan = db.get_plan(plan_id)
        if not plan:
            return jsonify({'error': f'Plan with ID {plan_id} not found'}), 404
        
        # Run validation and refinement workflow
        result = team_coordinator.execute_validation_and_refinement_workflow(plan)
        
        return jsonify(result)
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """System health check endpoint"""
    try:
        health_results = health_monitor.run_health_checks()
        return jsonify(health_results)
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        # Get database stats
        db_stats = db.get_database_stats()
        
        # Get error stats
        error_stats = error_handler.get_error_stats()
        
        return jsonify({
            'database': db_stats,
            'errors': error_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/gauntlets/run', methods=['POST'])
def run_gauntlets():
    """Run gauntlets on a plan"""
    try:
        data = request.json
        plan_id = data.get('plan_id')
        
        plan = db.get_plan(plan_id)
        if not plan:
            return jsonify({'error': f'Plan with ID {plan_id} not found'}), 404
        
        # Run decomposition gauntlets
        results = gauntlet_system.run_decomposition_gauntlets(plan)
        
        return jsonify({
            'plan_id': plan_id,
            'results': {k: v.to_dict() for k, v in results.items()},
            'overall_quality': gauntlet_system.get_overall_quality(results),
            'all_passed': gauntlet_system.all_passed(results)
        })
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/subproblems/<subproblem_id>/solution', methods=['POST'])
def submit_solution(subproblem_id):
    """Submit a solution for a sub-problem"""
    try:
        data = request.json
        approach = data.get('approach', '')
        solution_content = data.get('solution_content', '')
        team_id = data.get('team_id', 'default')
        confidence_score = data.get('confidence_score', 0.8)
        
        # Get sub-problem
        sub_problem = db.get_subproblem(subproblem_id)
        if not sub_problem:
            return jsonify({'error': f'Sub-problem with ID {subproblem_id} not found'}), 404
        
        # Create solution attempt
        attempt = solution_orchestrator.track_solution_attempt(
            subproblem_id, approach, solution_content, team_id, confidence_score
        )
        
        # Validate solution
        validation_result = solution_orchestrator.validate_solution(attempt, sub_problem)
        
        return jsonify({
            'attempt': attempt.to_dict(),
            'validation': validation_result.to_dict()
        })
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/plans/<plan_id>/integrate', methods=['POST'])
def integrate_solutions(plan_id):
    """Integrate solutions for a plan"""
    try:
        plan = db.get_plan(plan_id)
        if not plan:
            return jsonify({'error': f'Plan with ID {plan_id} not found'}), 404
        
        # Integrate solutions
        integrated_solution = solution_orchestrator.integrate_solutions(plan)
        
        return jsonify(integrated_solution.to_dict())
    except Exception as e:
        error_handler.handle_error(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Register health checks
    from sovereign_reliability import register_core_health_checks
    register_core_health_checks()
    
    # Start the server
    app.run(host='127.0.0.1', port=8081, debug=True)