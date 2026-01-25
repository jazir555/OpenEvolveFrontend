# Team Assignment Engine - Quick Reference

## Quick Start

```python
from team_assignment_engine import TeamAssignmentEngine
from team_manager import TeamManager

# Initialize
team_manager = TeamManager()
engine = TeamAssignmentEngine(team_manager)

# Assign teams to a plan
teams = team_manager.get_all_teams()
plan_with_teams = engine.assign_teams_to_plan(decomposition_plan, teams)
```

## Key Concepts

### TeamCapability
- **Overall Score**: Weighted combination of capability, success rate, workload, and specialization
- **Weights**: 35% capability, 30% success rate, 20% workload, 15% specialization

### Assignment Roles
- **Solver**: Blue team for core development
- **Patcher**: Blue team for refinement (often same as solver)
- **Red Team**: Red team for adversarial critique
- **Gold Team**: Gold team for verification

### Conflict Avoidance
- Solver and Red Team should be different teams
- Ensures adversarial perspective
- Falls back to same team only if necessary

## Common Operations

### Assess Team Capability
```python
assessor = TeamCapabilityAssessor(team_manager)
capability = assessor.assess_team_capability(team, sub_problem)
print(f"Overall capability: {capability.calculate_overall_capability():.2f}")
```

### Assign Single Sub-Problem
```python
assignment = engine.assign_teams_to_subproblem(sub_problem, teams)
print(f"Solver: {assignment.solver}")
print(f"Red Team: {assignment.red_team}")
```

### Track Performance
```python
tracker = TeamPerformanceTracker()

# Record assignment
tracker.record_assignment(team_id, sub_problem_id, "solver", assignment)

# Record outcome
tracker.record_outcome(team_id, sub_problem_id, True, 0.92, 180.5)

# Get stats
stats = tracker.get_team_performance_stats(team_id)
print(f"Success rate: {stats['success_rate']:.2%}")
```

### Get Rankings
```python
rankings = tracker.get_team_ranking()
for team_id, score in rankings[:5]:
    print(f"{team_id}: {score:.2f}")
```

## Integration with DecompositionEngine

```python
from decomposition_engine import DecompositionEngine
from team_assignment_engine import TeamAssignmentEngine

# Create assignment engine
assignment_engine = TeamAssignmentEngine(team_manager)

# Create decomposition engine with team assignment
decomp_engine = DecompositionEngine(
    team_assignment_engine=assignment_engine
)

# Decompose with team assignment
plan = decomp_engine.decompose(
    problem=problem_definition,
    assign_teams=True,
    teams=teams
)
```

## Team Specialization

```python
team = Team(
    name="Security-Experts",
    role="Blue",
    members=[...],
    domain_specialization=["security", "authentication"],
    problem_type_specialization=["implementation"],
    performance_metrics={"accuracy": 0.90}
)
```

## Weights and Scoring

### Capability Score
- Expertise matching: 40%
- Team role: 20%
- Performance metrics: 30%
- Team configuration: 10%

### Overall Capability
- Capability score: 35%
- Success rate: 30%
- Workload: 20% (inverted - lower is better)
- Specialization fit: 15%

### Assignment Confidence
- Capability match: 40%
- Historical performance: 30%
- Workload availability: 20%
- Specialization fit: 10%

## Testing

```bash
# Run all tests
pytest test_team_assignment.py -v

# Run with coverage
pytest test_team_assignment.py --cov=team_assignment_engine
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No teams assigned | Check `assign_teams=True` and teams list is not empty |
| Poor assignments | Verify team `domain_specialization` and `performance_metrics` |
| Low confidence | Ensure teams have historical performance data |
| Data not persisting | Check file permissions and storage path |

## File Locations

- Implementation: `team_assignment_engine.py`
- Tests: `test_team_assignment.py`
- Documentation: `TEAM_ASSIGNMENT_COMPLETE.md`
- This guide: `TEAM_ASSIGNMENT_QUICK_REFERENCE.md`
