# Team Assignment Engine - Complete Implementation

## Overview

The Team Assignment Engine provides intelligent, multi-factor team assignment to sub-problems in the decomposition workflow. It analyzes team capabilities, historical performance, workload, and specialization to recommend optimal team assignments for solver, patcher, red team, and gold team roles.

## Architecture

### Core Components

```
TeamAssignmentEngine
├── TeamCapabilityAssessor
│   ├── Expertise Matching
│   ├── Capability Scoring
│   ├── Workload Assessment
│   └── Specialization Fit
├── TeamPerformanceTracker
│   ├── Assignment Recording
│   ├── Outcome Tracking
│   ├── Performance Statistics
│   └── Team Ranking
└── TeamCapability
    ├── Overall Capability Score
    ├── Success Rate
    ├── Recent Performance
    └── Confidence Score
```

### Data Flow

```
1. Sub-Problem Definition
   ↓
2. Team Capability Assessment
   ├── Expertise Matching (40%)
   ├── Historical Performance (30%)
   ├── Workload Availability (20%)
   └── Specialization Fit (10%)
   ↓
3. Team Assignment
   ├── Solver Assignment (Blue teams)
   ├── Patcher Assignment (Blue teams)
   ├── Red Team Assignment (Red teams)
   └── Gold Team Assignment (Gold teams)
   ↓
4. Performance Tracking
   ├── Record Assignment
   ├── Record Outcome
   └── Update Statistics
```

## Implementation Details

### 1. TeamCapability Class

Represents a team's capability assessment for a specific sub-problem.

**Attributes:**
- `team_id`: Unique team identifier
- `team_name`: Human-readable team name
- `domain`: Domain of expertise
- `expertise_areas`: List of specific expertise areas
- `capability_score`: Base capability score (0.0-1.0)
- `success_rate`: Historical success rate (0.0-1.0)
- `total_assignments`: Total number of past assignments
- `recent_performance`: Last 10 assignment results
- `workload_score`: Current workload (0.0-1.0, lower is better)
- `specialization_fit`: How well team specialization matches (0.0-1.0)
- `confidence_score`: Overall confidence in assessment (0.0-1.0)

**Methods:**
- `calculate_overall_capability()`: Combines all factors into single score

**Calculation:**
```python
overall = (
    0.35 * capability_score +
    0.30 * success_rate +
    0.20 * (1.0 - workload_score) +  # Inverted workload
    0.15 * specialization_fit
)
```

### 2. TeamCapabilityAssessor Class

Assesses team capabilities for sub-problems.

**Methods:**

#### `assess_team_capability(team, sub_problem) -> TeamCapability`

Assesses how capable a team is for a specific sub-problem.

**Factors Considered:**
1. **Expertise Matching** (40%)
   - Compares team's `domain_specialization` with sub-problem's `required_expertise`
   - Checks team's `problem_type_specialization`
   - Returns matched expertise areas and match score

2. **Capability Scoring** (Base)
   - Team role appropriateness (Blue for solving)
   - Performance metrics from team history
   - Team configuration quality

3. **Historical Performance** (30%)
   - Retrieves from TeamManager's aggregated metrics
   - Uses `avg_fitness` as success rate
   - Falls back to 0.5 if no data available

4. **Workload Assessment** (20%)
   - Based on `total_operations` in team metrics
   - More operations = higher workload
   - Simple heuristic: `min(1.0, total_ops / 100.0)`

5. **Specialization Fit** (10%)
   - Checks if team's specialization matches sub-problem description
   - Searches for domain keywords in title and description
   - Returns ratio of matches to total specializations

#### `assess_all_teams(sub_problem, available_teams) -> Dict[str, TeamCapability]`

Assesses all available teams for a sub-problem.

**Returns:**
```python
{
    "team_name": TeamCapability,
    ...
}
```

### 3. TeamAssignmentEngine Class

Intelligently assigns teams to sub-problems.

**Methods:**

#### `assign_teams_to_subproblem(sub_problem, available_teams) -> SubProblemTeamAssignment`

Assigns teams to a single sub-problem.

**Assignment Logic:**

1. **Solver Assignment**
   - Filters for Blue teams only
   - Sorts by overall capability score
   - Returns highest-scoring team

2. **Patcher Assignment**
   - Defaults to solver team
   - Can be specialized if needed

3. **Red Team Assignment**
   - Filters for Red teams only
   - Excludes solver (conflict avoidance)
   - Sorts by overall capability
   - Returns highest-scoring available team

4. **Gold Team Assignment**
   - Filters for Gold teams only
   - Sorts by overall capability
   - Returns highest-scoring team

**Conflict Avoidance:**
- Solver and Red Team should be different teams
- Ensures adversarial perspective
- Falls back to same team only if necessary

#### `assign_teams_to_plan(decomposition_plan, available_teams) -> DecompositionPlan`

Assigns teams to all sub-problems in a decomposition plan.

**Optimizations:**
- Best overall team assignments
- Balanced workload across teams
- Specialization utilization
- Tracks team usage for load balancing

**Returns:**
- Updated DecompositionPlan with `ai_suggested_team_assignment` populated for all sub-problems

#### `calculate_assignment_confidence(sub_problem, team) -> float`

Calculates confidence score for a team assignment (0.0-1.0).

**Factors:**
- Capability match (40%)
- Historical performance (30%)
- Workload availability (20%)
- Specialization fit (10%)

### 4. TeamPerformanceTracker Class

Tracks team performance over time for better assignments.

**Methods:**

#### `record_assignment(team_id, sub_problem_id, role, assignment)`

Records a team assignment for tracking.

**Updates:**
- Team's total assignments count
- Assignments by role
- Last assigned timestamp
- Domain expertise tracking

#### `record_outcome(team_id, sub_problem_id, success, quality_score, time_taken)`

Records the outcome of a team's work.

**Updates:**
- Successful assignments count
- Quality scores history
- Time taken history
- Success rate calculation

#### `get_team_performance_stats(team_id) -> Dict[str, Any]`

Get performance statistics for a team.

**Returns:**
```python
{
    'team_id': str,
    'total_assignments': int,
    'success_rate': float,
    'average_quality_score': float,
    'average_time_taken': float,
    'best_domains': List[str],
    'recent_performance_trend': float,
    'assignments_by_role': Dict[str, int],
    'first_assigned': str (ISO timestamp),
    'last_assigned': str (ISO timestamp)
}
```

#### `get_team_ranking(domain=None) -> List[Tuple[str, float]]`

Get teams ranked by performance.

**Calculation:**
```python
score = (
    0.5 * success_rate +
    0.3 * average_quality_score +
    0.2 * recent_performance_trend
)
```

**Domain Filtering:**
- If domain specified, boosts score for teams good in that domain
- Penalizes teams without domain expertise

**Returns:**
- List of (team_id, score) tuples, sorted by score descending

#### `get_performance_summary() -> Dict[str, Any]`

Get overall performance summary across all teams.

**Returns:**
```python
{
    'total_teams': int,
    'total_assignments': int,
    'total_outcomes': int,
    'overall_success_rate': float,
    'top_performing_teams': List[Dict],
    'metadata': Dict
}
```

## Usage Examples

### Basic Usage

```python
from team_assignment_engine import (
    TeamAssignmentEngine,
    TeamPerformanceTracker
)
from team_manager import TeamManager
from sovereign_data_models import DecompositionPlan

# Initialize components
team_manager = TeamManager()
performance_tracker = TeamPerformanceTracker(storage_path="team_performance.json")

# Create assignment engine
engine = TeamAssignmentEngine(
    team_manager=team_manager,
    performance_tracker=performance_tracker
)

# Get available teams
teams = team_manager.get_all_teams()

# Assign teams to decomposition plan
plan = ...  # Your DecompositionPlan
plan_with_teams = engine.assign_teams_to_plan(plan, teams)

# Access assignments
for sub_problem in plan_with_teams.sub_problems:
    assignment = sub_problem.ai_suggested_team_assignment
    print(f"Sub-problem: {sub_problem.title}")
    print(f"  Solver: {assignment.solver}")
    print(f"  Red Team: {assignment.red_team}")
    print(f"  Gold Team: {assignment.gold_team}")
```

### With Decomposition Engine

```python
from decomposition_engine import DecompositionEngine
from team_assignment_engine import TeamAssignmentEngine
from team_manager import TeamManager

# Initialize
team_manager = TeamManager()
assignment_engine = TeamAssignmentEngine(team_manager)

# Create decomposition engine with team assignment
decomposition_engine = DecompositionEngine(
    team_assignment_engine=assignment_engine
)

# Decompose with team assignment
teams = team_manager.get_all_teams()
plan = decomposition_engine.decompose(
    problem=problem_definition,
    assign_teams=True,
    teams=teams
)

# Plan now has team assignments
```

### Recording Outcomes

```python
# After a sub-problem is completed
performance_tracker.record_outcome(
    team_id="Blue-Security",
    sub_problem_id="sub_001",
    success=True,
    quality_score=0.92,
    time_taken=180.5
)

# Get updated stats
stats = performance_tracker.get_team_performance_stats("Blue-Security")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average quality: {stats['average_quality_score']:.2f}")
```

### Getting Team Rankings

```python
# Get overall rankings
rankings = performance_tracker.get_team_ranking()
for team_id, score in rankings:
    print(f"{team_id}: {score:.2f}")

# Get rankings for specific domain
security_rankings = performance_tracker.get_team_ranking(domain="security")
```

## Integration with Decomposition Workflow

### LLM Prompt Enhancement

The decomposition LLM prompt now includes detailed team assignment guidance:

```
Team_Assignment: [Recommend which team should handle this based on:
- Solver: Match required expertise to team specialization
- Patcher: May be same as solver or different specialized team
- Red_Team: Match domain expertise for critique
- Gold_Team: Match verification specialization]
```

This helps the LLM provide AI-suggested team assignments that are then refined by the TeamAssignmentEngine.

### SubProblem Team Assignment Field

Each `SubProblem` now has an `ai_suggested_team_assignment` field:

```python
@dataclass
class SubProblemTeamAssignment:
    solver: str = ""
    patcher: str = ""
    red_team: str = ""
    gold_team: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Configuration Options

### Team Specialization

Teams can specify their specialization:

```python
team = Team(
    name="Security-Specialists",
    role="Blue",
    members=[...],
    domain_specialization=["security", "authentication", "cryptography"],
    problem_type_specialization=["implementation", "validation"],
    performance_metrics={"accuracy": 0.90, "security_score": 0.95}
)
```

### Performance Tracking Configuration

```python
# Customize storage path
tracker = TeamPerformanceTracker(storage_path="custom/path/performance.json")

# Or use in-memory (no persistence)
tracker = TeamPerformanceTracker(storage_path=None)
```

### Assignment Engine Configuration

```python
# Use custom capability assessor
custom_assessor = MyCustomAssessor(team_manager)
engine = TeamAssignmentEngine(
    team_manager=team_manager,
    capability_assessor=custom_assessor,
    performance_tracker=tracker
)
```

## Performance Considerations

### Scalability

- **Capability Assessment**: O(T × S) where T = teams, S = sub-problems
- **Team Assignment**: O(T log T) per sub-problem (sorting)
- **Performance Tracking**: O(1) recording, O(T) for rankings

### Optimization Strategies

1. **Caching**: Team capabilities can be cached for similar sub-problems
2. **Batch Processing**: Assess all teams for all sub-problems in one pass
3. **Lazy Evaluation**: Only assess teams when needed
4. **Incremental Updates**: Update performance stats incrementally

### Memory Usage

- TeamCapability: ~1 KB per team per sub-problem
- Performance data: ~500 bytes per assignment/outcome
- Tracker storage: Grows with usage, consider periodic cleanup

## Testing

### Running Tests

```bash
# Run all tests
pytest test_team_assignment.py -v

# Run specific test
pytest test_team_assignment.py::test_assign_teams_to_subproblem -v

# Run with coverage
pytest test_team_assignment.py --cov=team_assignment_engine --cov-report=html
```

### Test Coverage

The test suite includes:

1. **Capability Assessment Tests**
   - `test_capability_assessor_initialization`
   - `test_assess_team_capability`
   - `test_assess_team_capability_expertise_match`
   - `test_assess_all_teams`

2. **Team Assignment Tests**
   - `test_assign_teams_to_subproblem`
   - `test_assign_teams_security_subproblem`
   - `test_assign_teams_general_subproblem`
   - `test_conflict_avoidance`
   - `test_calculate_assignment_confidence`
   - `test_assign_teams_to_plan`

3. **Performance Tracking Tests**
   - `test_record_assignment`
   - `test_record_outcome`
   - `test_get_team_performance_stats`
   - `test_get_team_ranking`
   - `test_get_performance_summary`
   - `test_performance_tracker_persistence`

4. **Integration Tests**
   - `test_full_assignment_workflow`

## Error Handling

The system includes comprehensive error handling:

1. **Graceful Degradation**
   - Missing team data → Returns default/neutral scores
   - Assessment failures → Returns low-capability defaults
   - Assignment failures → Returns empty assignment

2. **Logging**
   - All operations logged at appropriate levels
   - Error context preserved for debugging
   - Performance metrics tracked

3. **Fallback Behavior**
   - No teams available → Returns empty assignments
   - Single team → Assigns to all roles
   - Missing performance data → Uses neutral defaults

## Future Enhancements

### Planned Features

1. **Machine Learning Enhancement**
   - Learn optimal assignments from historical data
   - Predict team performance on new problems
   - Adaptive weight tuning

2. **Advanced Workload Balancing**
   - Real-time workload tracking
   - Predictive availability management
   - Team capacity planning

3. **Skill Matrix**
   - Fine-grained skill tracking
   - Multi-dimensional expertise
   - Dynamic skill assessment

4. **Team Composition Optimization**
   - Suggest optimal team member combinations
   - Balance team diversity and capability
   - Consider team dynamics

5. **Collaboration Patterns**
   - Track which teams work well together
   - Suggest team pairings
   - Learn from successful collaborations

## Troubleshooting

### Common Issues

**Issue: Teams not being assigned**
- Check that `assign_teams=True` is passed to `decompose()`
- Verify teams list is not empty
- Check team roles (Blue/Red/Gold)

**Issue: Poor team assignments**
- Verify team `domain_specialization` is set
- Check team `performance_metrics` have data
- Review sub-problem `required_expertise` field

**Issue: Performance data not persisting**
- Check file permissions for storage path
- Verify storage path is valid
- Check for disk space

**Issue: Low confidence scores**
- Ensure teams have historical performance data
- Verify expertise areas are specified
- Check workload calculations

## API Reference

### TeamAssignmentEngine

```python
class TeamAssignmentEngine:
    def __init__(
        self,
        team_manager: TeamManager,
        capability_assessor: Optional[TeamCapabilityAssessor] = None,
        performance_tracker: Optional[TeamPerformanceTracker] = None
    )

    def assign_teams_to_subproblem(
        self,
        sub_problem: SubProblem,
        available_teams: List[Team]
    ) -> SubProblemTeamAssignment

    def assign_teams_to_plan(
        self,
        decomposition_plan: DecompositionPlan,
        available_teams: List[Team]
    ) -> DecompositionPlan

    def calculate_assignment_confidence(
        self,
        sub_problem: SubProblem,
        team: Team
    ) -> float
```

### TeamPerformanceTracker

```python
class TeamPerformanceTracker:
    def __init__(self, storage_path: str = "team_performance.json")

    def record_assignment(
        self,
        team_id: str,
        sub_problem_id: str,
        role: str,
        assignment: SubProblemTeamAssignment
    )

    def record_outcome(
        self,
        team_id: str,
        sub_problem_id: str,
        success: bool,
        quality_score: float,
        time_taken: float
    )

    def get_team_performance_stats(
        self,
        team_id: str
    ) -> Dict[str, Any]

    def get_team_ranking(
        self,
        domain: Optional[str] = None
    ) -> List[Tuple[str, float]]

    def get_performance_summary(
        self
    ) -> Dict[str, Any]
```

### TeamCapabilityAssessor

```python
class TeamCapabilityAssessor:
    def __init__(self, team_manager: TeamManager)

    def assess_team_capability(
        self,
        team: Team,
        sub_problem: SubProblem
    ) -> TeamCapability

    def assess_all_teams(
        self,
        sub_problem: SubProblem,
        available_teams: List[Team]
    ) -> Dict[str, TeamCapability]
```

## Conclusion

The Team Assignment Engine provides a comprehensive, production-ready solution for intelligent team assignment in the decomposition workflow. It combines multiple factors to make optimal assignments, tracks performance for continuous improvement, and integrates seamlessly with the existing decomposition infrastructure.

For questions or issues, refer to the test suite for usage examples or consult the inline documentation in the source code.
