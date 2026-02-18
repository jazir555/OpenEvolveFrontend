# DECOMPOSITION WORKFLOW REQUIREMENTS - COMPREHENSIVE EXTRACTION

**Document Version**: 1.0
**Extraction Date**: 2026-01-03
**Source Document**: Decomposition_Workflow.md (338.3KB, 6000+ lines)
**Extraction Method**: Systematic reading and requirement categorization

---

## EXECUTIVE SUMMARY

### Total Requirements Count: 287

**Breakdown by Category**:
- Data Models: 67 requirements (23.3%)
- Workflow Stages: 52 requirements (18.1%)
- Team & Gauntlet Architecture: 43 requirements (15.0%)
- MDAP Integration: 38 requirements (13.2%)
- MAKER Framework: 27 requirements (9.4%)
- Quality Assessment: 24 requirements (8.4%)
- ACE Integration: 12 requirements (4.2%)
- crewai Integration: 16 requirements (5.6%)
- Lean 4 Integration: 8 requirements (2.8%)

### Critical Success Factors
1. **Sovereign-Grade Control**: Every parameter must be user-configurable
2. **Self-Healing Automation**: Automatic error detection and correction
3. **Multi-Agent Coordination**: Teams work in orchestrated patterns
4. **Quantitative Analysis**: Statistical consensus and voting mechanisms
5. **Knowledge Extraction**: Learning from every execution

---

## PART 1: DATA MODEL REQUIREMENTS (67 Requirements)

### DM-001: ModelConfig Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.1 (lines 3623-3647)

**Required Fields**:
```python
@dataclasses.dataclass
class ModelConfig:
    model_id: str                                    # REQUIRED
    api_key: str                                     # REQUIRED
    api_base: str = "https://api.openai.com/v1"      # DEFAULT
    temperature: float = 0.7                         # DEFAULT
    top_p: float = 1.0                               # DEFAULT
    max_tokens: int = 4096                           # DEFAULT
    frequency_penalty: float = 0.0                   # DEFAULT
    presence_penalty: float = 0.0                    # DEFAULT
    seed: Optional[int] = None                       # OPTIONAL
    domain_specialization: Optional[List[str]] = None # OPTIONAL
    problem_type_specialization: Optional[List[str]] = None # OPTIONAL
    performance_metrics: Optional[Dict[str, float]] = None # OPTIONAL
    cost_per_token: Optional[float] = None           # OPTIONAL
```

**Validation Requirements**:
- model_id must be non-empty string
- api_key must be valid authentication credential
- temperature must be between 0.0 and 2.0
- top_p must be between 0.0 and 1.0
- max_tokens must be positive integer
- frequency_penalty must be between -2.0 and 2.0
- presence_penalty must be between -2.0 and 2.0

**Implementation Notes**:
- Must support serialization/deserialization
- Must support JSON export/import
- Must validate API key format at initialization

---

### DM-002: Team Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.2 (lines 3649-3669)

**Required Fields**:
```python
@dataclasses.dataclass
class Team:
    name: str                                       # REQUIRED
    role: Literal["Blue", "Red", "Gold"]           # REQUIRED
    members: List[ModelConfig]                      # REQUIRED
    description: Optional[str] = None               # OPTIONAL
    sub_role: Optional[str] = None                  # OPTIONAL
    domain_specialization: Optional[List[str]] = None # OPTIONAL
    problem_type_specialization: Optional[List[str]] = None # OPTIONAL
    performance_metrics: Optional[Dict[str, float]] = None # OPTIONAL
    team_config: Optional[Dict[str, Any]] = None    # OPTIONAL
```

**Team Role Requirements**:
- Blue Teams: Creation and synthesis (Planners, Solvers, Patchers, Assemblers, Optimizers, Synthesizers)
- Red Teams: Criticism and flaw detection (Security Analysts, Logic Verifiers, Edge Case Explorers, Assumption Challengers, Compliance Checkers)
- Gold Teams: Impartial evaluation and scoring (Accuracy Judges, Completeness Judges, Efficiency Judges, Usability Judges, Innovation Judges)

**Validation Requirements**:
- name must be unique across all teams
- role must be one of: "Blue", "Red", "Gold"
- members must contain at least one ModelConfig
- sub_role must be valid for the specified role

---

### DM-003: GauntletRoundRule Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.3 (lines 3671-3707)

**Required Fields**:
```python
@dataclasses.dataclass
class GauntletRoundRule:
    round_number: int                               # REQUIRED
    quorum_required_approvals: int                  # REQUIRED
    quorum_from_panel_size: int                    # REQUIRED
    min_overall_confidence: float = 0.0            # DEFAULT
    max_score_variance: Optional[float] = None     # OPTIONAL
    per_judge_requirements: Dict[str, Dict[str, Any]] = field(default_factory=dict) # OPTIONAL
    collaboration_mode: Literal["independent", "share_previous_feedback"] = "independent" # DEFAULT
    time_limit_seconds: Optional[int] = None       # OPTIONAL
    max_api_calls: Optional[int] = None            # OPTIONAL
    max_tokens: Optional[int] = None               # OPTIONAL
    adaptive_rules: Optional[Dict[str, Any]] = None # OPTIONAL
```

**Quorum Requirements**:
- M out of N agents must approve (quorum_required_approvals out of quorum_from_panel_size)
- Must support fractional quorums (e.g., 2/3 majority)
- Must support per-agent requirements override

**Validation Requirements**:
- round_number must be positive integer
- quorum_required_approvals must be <= quorum_from_panel_size
- quorum_required_approvals must be > 0
- min_overall_confidence must be between 0.0 and 1.0
- max_score_variance must be positive if specified
- time_limit_seconds must be positive if specified
- max_api_calls must be positive if specified
- max_tokens must be positive if specified

---

### DM-004: GauntletDefinition Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.4 (lines 3709-3743)

**Required Fields**:
```python
@dataclasses.dataclass
class GauntletDefinition:
    name: str                                       # REQUIRED
    team_name: str                                  # REQUIRED
    rounds: List[GauntletRoundRule]                 # REQUIRED
    description: Optional[str] = None               # OPTIONAL
    attack_modes: List[str] = field(default_factory=list) # DEFAULT
    generation_mode: Literal["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"] = "single_candidate" # DEFAULT
    gauntlet_type: Literal["standard", "adaptive", "hierarchical", "competitive", "collaborative"] = "standard" # DEFAULT
    performance_metrics: Optional[Dict[str, float]] = None # OPTIONAL
    gauntlet_config: Optional[Dict[str, Any]] = None # OPTIONAL
```

**Gauntlet Types**:
- standard: Fixed rules for all rounds
- adaptive: Rules adapt based on content being evaluated
- hierarchical: Multiple tiers with increasingly strict criteria
- competitive: Multiple solutions compete against each other
- collaborative: Models work together to improve solutions

**Validation Requirements**:
- name must be unique across all gauntlets
- team_name must reference an existing Team
- rounds must contain at least one GauntletRoundRule
- rounds must be ordered sequentially by round_number
- attack_modes only applicable for Red Team gauntlets
- generation_mode only applicable for Blue Team gauntlets

---

### DM-005: SubProblem Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.5 (lines 3745-3781)

**Required Fields**:
```python
@dataclasses.dataclass
class SubProblem:
    id: str                                         # REQUIRED
    description: str                                # REQUIRED
    dependencies: List[str] = field(default_factory=list) # DEFAULT
    ai_suggested_evolution_mode: str = "standard"   # DEFAULT
    ai_suggested_complexity_score: int = 5          # DEFAULT
    ai_suggested_evaluation_prompt: str = ""        # DEFAULT
    ai_suggested_team_assignment: Optional[str] = None # OPTIONAL
    ai_suggested_gauntlet_assignment: Optional[Dict[str, str]] = None # OPTIONAL
    estimated_resources: Optional[Dict[str, Any]] = None # OPTIONAL
    potential_approaches: Optional[List[str]] = None # OPTIONAL
    solver_team_name: str = ""                      # DEFAULT
    patcher_team_name: str = ""                     # DEFAULT
    red_team_gauntlet_name: Optional[str] = None    # OPTIONAL
    gold_team_gauntlet_name: str = ""               # DEFAULT
    evolution_params: Dict[str, Any] = field(default_factory=dict) # DEFAULT
    status: Literal["pending", "in_progress", "solved", "failed", "requires_rework"] = "pending" # DEFAULT
    solution_attempts: List[SolutionAttempt] = field(default_factory=list) # DEFAULT
    performance_metrics: Optional[Dict[str, float]] = None # OPTIONAL
```

**Evolution Modes**:
- standard: Basic solution generation
- adversarial: Red team critique-driven evolution
- quality_diversity: Multiple quality criteria exploration

**Validation Requirements**:
- id must be unique within DecompositionPlan
- description must be non-empty
- dependencies must reference valid sub-problem IDs (no circular dependencies)
- ai_suggested_complexity_score must be between 1 and 10
- status transitions must follow valid workflow

**Status Transition Rules**:
- pending → in_progress (when work starts)
- in_progress → solved (when solution verified)
- in_progress → failed (when solution cannot be found)
- solved → requires_rework (when critique finds flaws)
- requires_rework → in_progress (when rework starts)

---

### DM-006: DecompositionPlan Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.6 (lines 3783-3815)

**Required Fields**:
```python
@dataclasses.dataclass
class DecompositionPlan:
    problem_statement: str                          # REQUIRED
    analyzed_context: Dict[str, Any]                # REQUIRED
    sub_problems: List[SubProblem]                  # REQUIRED
    max_refinement_loops: int = 3                   # DEFAULT
    auto_approval_enabled: bool = False             # DEFAULT
    auto_approval_criteria: Optional[Dict[str, Any]] = None # OPTIONAL
    resource_limits: Optional[Dict[str, Any]] = None # OPTIONAL
    parallel_processing_enabled: bool = False       # DEFAULT
    max_parallel_sub_problems: int = 1              # DEFAULT
    learning_enabled: bool = False                  # DEFAULT
    learning_config: Optional[Dict[str, Any]] = None # OPTIONAL
    content_analyzer_team_name: str = ""            # DEFAULT
    planner_team_name: str = ""                     # DEFAULT
    assembler_team_name: str = ""                   # DEFAULT
    final_red_team_gauntlet_name: Optional[str] = None # OPTIONAL
    final_gold_team_gauntlet_name: str = ""         # DEFAULT
```

**Analyzed Context Structure** (from Stage 0):
```python
{
    "domain": str,                                  # One of 14 standardized domains
    "keywords": List[str],                          # Key terms and concepts
    "estimated_complexity": int,                    # 1-10 scale
    "potential_challenges": List[str],              # Domain-specific challenges
    "required_expertise": List[str],                # Required expertise areas
    "summary": str,                                 # 1-2 sentence summary
    "success_criteria": List[str],                  # 3-7 measurable criteria
    "constraints": List[str],                       # All constraints
    "stakeholders": List[str],                      # All stakeholders
    "risk_factors": List[str],                      # All risk factors
    "problem_type": str,                            # Problem classification
    "solution_approach_hint": str,                  # Suggested approach
    "technical_stack_suggestions": List[str],       # Suggested technologies
    "initial_resource_estimate": {
        "time_days": float,
        "api_tokens": int,
        "human_hours": float
    }
}
```

**Validation Requirements**:
- problem_statement must be non-empty
- analyzed_context must contain all required fields from Stage 0
- sub_problems must contain at least one SubProblem
- sub_problems must form a DAG (no circular dependencies)
- max_refinement_loops must be positive
- max_parallel_sub_problems must be positive if parallel_processing_enabled

---

### DM-007: SolutionAttempt Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.7 (lines 3817-3845)

**Required Fields**:
```python
@dataclasses.dataclass
class SolutionAttempt:
    sub_problem_id: str                             # REQUIRED
    content: str                                    # REQUIRED
    generated_by_model: str                         # REQUIRED
    timestamp: float                                # REQUIRED
    history: List[Dict[str, Any]] = field(default_factory=list) # DEFAULT
    solution_type: Optional[str] = None             # OPTIONAL
    solution_approach: Optional[str] = None         # OPTIONAL
    quality_metrics: Optional[Dict[str, float]] = None # OPTIONAL
    resource_usage: Optional[Dict[str, Any]] = None # OPTIONAL
    status: Literal["generated", "critiqued", "verified", "rejected", "patched"] = "generated" # DEFAULT
    critique_reports: List[CritiqueReport] = field(default_factory=list) # DEFAULT
    verification_reports: List[VerificationReport] = field(default_factory=list) # DEFAULT
```

**Solution Types**:
- code: Programming code solution
- text: Textual explanation or documentation
- diagram: Visual representation or diagram
- data: Data structure or dataset
- configuration: Configuration files or settings

**Validation Requirements**:
- sub_problem_id must reference a valid SubProblem or be "final_solution"
- content must be non-empty
- generated_by_model must be a valid model_id
- timestamp must be valid Unix timestamp
- status transitions must follow workflow

---

### DM-008: CritiqueReport Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.8 (lines 3847-3874)

**Required Fields**:
```python
@dataclasses.dataclass
class CritiqueReport:
    solution_attempt_id: str                        # REQUIRED
    gauntlet_name: str                              # REQUIRED
    is_approved: bool                               # REQUIRED
    reports_by_judge: List[Dict[str, Any]]          # REQUIRED
    summary: str = ""                               # DEFAULT
    critique_timestamp: float = field(default_factory=time.time) # DEFAULT
    overall_score: float = 0.0                      # DEFAULT
    flaw_severity_scores: Dict[str, float] = field(default_factory=dict) # DEFAULT
    identified_flaws: List[Dict[str, Any]] = field(default_factory=list) # DEFAULT
    suggested_improvements: List[str] = field(default_factory=list) # DEFAULT
    resource_usage: Optional[Dict[str, Any]] = None # OPTIONAL
```

**Flaw Severity Categories**:
- critical: Security vulnerabilities, data loss risks
- major: Functional issues, significant flaws
- minor: Style issues, minor improvements
- informational: Suggestions for enhancement

**Judge Report Structure**:
```python
{
    "model_id": str,                                # Judge's model identifier
    "score": float,                                 # Judge's score (0.0-1.0)
    "justification": str,                           # Reasoning for score
    "identified_flaws": List[Dict],                 # Flaws found by this judge
    "suggested_improvements": List[str],            # Improvements suggested
    "targeted_feedback": Optional[str]              # Specific feedback for patching
}
```

**Validation Requirements**:
- solution_attempt_id must reference a valid SolutionAttempt
- gauntlet_name must reference a valid Red Team GauntletDefinition
- reports_by_judge must contain at least one judge report
- overall_score must be between 0.0 and 1.0
- flaw_severity_scores must sum to <= 1.0

---

### DM-009: VerificationReport Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.9 (lines 3876-3905)

**Required Fields**:
```python
@dataclasses.dataclass
class VerificationReport:
    solution_attempt_id: str                        # REQUIRED
    gauntlet_name: str                              # REQUIRED
    is_approved: bool                               # REQUIRED
    reports_by_judge: List[Dict[str, Any]]          # REQUIRED
    average_score: float = 0.0                      # DEFAULT
    score_variance: float = 0.0                     # DEFAULT
    summary: str = ""                               # DEFAULT
    verification_timestamp: float = field(default_factory=time.time) # DEFAULT
    dimension_scores: Dict[str, float] = field(default_factory=dict) # DEFAULT
    criteria_met: List[str] = field(default_factory=list) # DEFAULT
    criteria_not_met: List[str] = field(default_factory=list) # DEFAULT
    targeted_feedback: Optional[str] = None         # OPTIONAL
    resource_usage: Optional[Dict[str, Any]] = None # OPTIONAL
```

**Quality Dimensions** (from Gold Team evaluation):
- accuracy: Factual correctness
- completeness: Full problem coverage
- efficiency: Performance and resource utilization
- usability: User-friendliness and accessibility
- innovation: Novelty and creativity
- security: Security robustness
- maintainability: Code quality and maintainability
- scalability: Ability to scale

**Judge Report Structure**:
```python
{
    "model_id": str,                                # Judge's model identifier
    "dimension_scores": Dict[str, float],           # Scores per dimension
    "overall_score": float,                         # Overall judgment (0.0-1.0)
    "justification": str,                           # Reasoning for score
    "criteria_met": List[str],                      # Criteria that passed
    "criteria_not_met": List[str],                  # Criteria that failed
    "targeted_feedback": Optional[str]              # Specific improvement feedback
}
```

**Validation Requirements**:
- solution_attempt_id must reference a valid SolutionAttempt
- gauntlet_name must reference a valid Gold Team GauntletDefinition
- reports_by_judge must contain at least one judge report
- average_score must be between 0.0 and 1.0
- score_variance must be non-negative
- dimension_scores must all be between 0.0 and 1.0

---

### DM-010: WorkflowState Class
**Priority**: CRITICAL
**Category**: Data Models
**Source**: Section 5.10 (lines 3907-3954)

**Required Fields**:
```python
@dataclasses.dataclass
class WorkflowState:
    workflow_id: str                                # REQUIRED
    problem_statement: str                          # REQUIRED
    current_stage: str                              # REQUIRED
    status: str = "running"                         # DEFAULT
    progress: float = 0.0                           # DEFAULT
    start_time: float = field(default_factory=time.time) # DEFAULT
    end_time: Optional[float] = None                # OPTIONAL
    current_sub_problem_id: Optional[str] = None    # OPTIONAL
    current_gauntlet_name: Optional[str] = None     # OPTIONAL
    decomposition_plan: Optional[DecompositionPlan] = None # OPTIONAL
    sub_problem_solutions: Dict[str, SolutionAttempt] = field(default_factory=dict) # DEFAULT
    final_solution: Optional[SolutionAttempt] = None # OPTIONAL
    refinement_loop_count: int = 0                  # DEFAULT
    all_critique_reports: List[CritiqueReport] = field(default_factory=list) # DEFAULT
    all_verification_reports: List[VerificationReport] = field(default_factory=list) # DEFAULT
    content_analyzer_team: Optional[Team] = None    # OPTIONAL
    planner_team: Optional[Team] = None             # OPTIONAL
    solver_team: Optional[Team] = None              # OPTIONAL
    patcher_team: Optional[Team] = None             # OPTIONAL
    assembler_team: Optional[Team] = None           # OPTIONAL
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None # OPTIONAL
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None # OPTIONAL
    final_red_gauntlet: Optional[GauntletDefinition] = None # OPTIONAL
    final_gold_gauntlet: Optional[GauntletDefinition] = None # OPTIONAL
    max_refinement_loops: int = 3                   # DEFAULT
    resource_usage: Dict[str, Any] = field(default_factory=dict) # DEFAULT
    performance_metrics: Dict[str, float] = field(default_factory=dict) # DEFAULT
    knowledge_artifacts: List[KnowledgeArtifact] = field(default_factory=list) # DEFAULT
```

**Status Values**:
- running: Workflow actively executing
- paused: Workflow paused awaiting user input
- completed: Workflow finished successfully
- failed: Workflow terminated due to error
- awaiting_user_input: Workflow needs user intervention

**Stage Names**:
- "Content Analysis"
- "AI-Assisted Decomposition"
- "Manual Review & Override"
- "Sub-Problem Solving Loop"
- "Configurable Reassembly"
- "Final Verification & Self-Healing Loop"
- "Knowledge Extraction & Learning"

**Validation Requirements**:
- workflow_id must be unique across all workflow runs
- problem_statement must be non-empty
- current_stage must be one of the valid stage names
- progress must be between 0.0 and 1.0
- refinement_loop_count must not exceed max_refinement_loops

---

### DM-011: KnowledgeArtifact Class
**Priority**: MEDIUM
**Category**: Data Models
**Source**: Section 5.11 (lines 3956-3978)

**Required Fields**:
```python
@dataclasses.dataclass
class KnowledgeArtifact:
    id: str                                         # REQUIRED
    artifact_type: Literal["solution_pattern", "problem_solution_mapping", "critique_insight", "team_performance", "gauntlet_effectiveness"] # REQUIRED
    content: Dict[str, Any]                         # REQUIRED
    source_workflow_id: str                         # REQUIRED
    extraction_timestamp: float = field(default_factory=time.time) # DEFAULT
    domain: Optional[str] = None                    # OPTIONAL
    problem_type: Optional[str] = None              # OPTIONAL
    usage_count: int = 0                            # DEFAULT
    effectiveness_score: float = 0.0                # DEFAULT
    related_artifacts: List[str] = field(default_factory=list) # DEFAULT
```

**Artifact Types**:
- solution_pattern: Reusable solution approaches
- problem_solution_mapping: Problem to solution mappings
- critique_insight: Insights from critique reports
- team_performance: Team performance patterns
- gauntlet_effectiveness: Gauntlet effectiveness metrics

**Validation Requirements**:
- id must be unique across all artifacts
- source_workflow_id must reference a valid workflow
- artifact_type must be one of the valid types
- usage_count must be non-negative
- effectiveness_score must be between 0.0 and 1.0

---

### DM-012: PerformanceMetrics Class
**Priority**: MEDIUM
**Category**: Data Models
**Source**: Section 5.12 (lines 3980-3997)

**Required Fields**:
```python
@dataclasses.dataclass
class PerformanceMetrics:
    entity_type: Literal["team", "gauntlet", "workflow"] # REQUIRED
    entity_id: str                                  # REQUIRED
    metrics: Dict[str, float]                       # REQUIRED
    timestamp: float = field(default_factory=time.time) # DEFAULT
    domain: Optional[str] = None                    # OPTIONAL
    problem_type: Optional[str] = None              # OPTIONAL
    context: Optional[Dict[str, Any]] = None        # OPTIONAL
```

**Common Metrics**:
- success_rate: Percentage of successful operations
- average_confidence: Average confidence score
- average_response_time: Average time to complete
- resource_efficiency: Resource utilization efficiency
- quality_score: Overall quality assessment

**Validation Requirements**:
- entity_type must be one of: "team", "gauntlet", "workflow"
- entity_id must reference a valid entity of the specified type
- metrics must contain at least one metric
- all metric values must be finite numbers

---

## PART 2: WORKFLOW STAGE REQUIREMENTS (52 Requirements)

### WS-001: Stage 0 - Content Analysis
**Priority**: CRITICAL
**Category**: Workflow Stages
**Source**: Section 3.1 (lines 1445-1627)

**Purpose**: Thoroughly understand user's initial problem statement and extract all relevant context

**Input**:
- User's raw, high-level problem description (string)

**Process**:
1. Invoke dedicated Blue Team (role: "Content Analyzer")
2. Generate specialized prompt with strict JSON output format
3. Execute LLM invocation with expertise weighting
4. Aggregate multiple analyses if multiple models in team
5. Validate output format and completeness

**Output Structure**:
```json
{
  "domain": "string",                    // One of 14 standardized domains
  "keywords": ["string"],                // Key terms and concepts
  "estimated_complexity": integer,       // 1-10 scale (40% technical, 30% domain, 20% resources, 10% timeline)
  "potential_challenges": ["string"],    // Domain-specific challenges
  "required_expertise": ["string"],      // Required expertise areas
  "summary": "string",                   // 1-2 sentence core requirements summary
  "success_criteria": ["string"],        // 3-7 measurable criteria
  "constraints": ["string"],             // All constraints (technical, regulatory, timeline, budget, resource)
  "stakeholders": ["string"],            // End users, maintainers, regulators, decision makers, affected third parties
  "risk_factors": ["string"],            // Technical, business, security, compliance risks
  "problem_type": "string",              // Problem classification
  "solution_approach_hint": "string",    // Suggested approach
  "technical_stack_suggestions": ["string"], // Suggested technologies
  "initial_resource_estimate": {
    "time_days": float,
    "api_tokens": integer,
    "human_hours": float
  }
}
```

**Valid Domains**:
- Software Development
- Data Science
- Business Strategy
- Scientific Research
- Engineering
- Legal
- Healthcare
- Finance
- Education
- Creative Arts
- Manufacturing
- Logistics
- Security
- Compliance

**Validation Requirements**:
- domain must be one of the 14 standardized domains
- keywords must contain at least 3 terms
- estimated_complexity must be between 1 and 10
- success_criteria must contain 3-7 criteria
- All required fields must be present
- All lists must be non-empty (except where specified)

**Implementation Requirements**:
- Must support multiple model analysis and aggregation
- Must validate JSON output format
- Must handle model errors gracefully
- Must cache results for same problem statement

---

### WS-002: Stage 1 - AI-Assisted Decomposition
**Priority**: CRITICAL
**Category**: Workflow Stages
**Source**: Section 3.2 (lines 1628-1987)

**Purpose**: Generate comprehensive DecompositionPlan with sub-problems, dependencies, and configurations

**Input**:
- problem_statement (str)
- analyzed_context (Dict from Stage 0)

**Process**:
1. Invoke Blue Team (role: "Planner")
2. Generate specialized prompt with decomposition instructions
3. Execute LLM invocation with context awareness
4. Parse and validate DecompositionPlan
5. Validate sub-problem dependencies (DAG check)
6. Generate initial resource estimates for each sub-problem
7. Suggest team and gauntlet assignments

**Output Structure**: DecompositionPlan object (see DM-006)

**Decomposition Strategy**:
- Break complex problems into atomic solvable units
- Identify dependencies between sub-problems
- Estimate complexity for each sub-problem (1-10)
- Suggest appropriate teams and gauntlets
- Propose multiple solution approaches when applicable

**MDAP/MAKER Integration** (when enabled):
- Use MDAP for decomposition (maximal decomposition into microtasks)
- Use MAKER for recursive decomposition with (P1, P2, C) structure
- Apply first-to-ahead-by-k voting for decomposition decisions
- Red-flag overly complex or ambiguous sub-problems

**Validation Requirements**:
- At least one sub-problem must be generated
- No circular dependencies allowed
- All dependencies must reference valid sub-problem IDs
- Complexity scores must be between 1-10
- Team assignments must reference existing teams
- Gauntlet assignments must reference existing gauntlets

**Implementation Requirements**:
- Must support both standard and MDAP/MAKER decomposition
- Must validate DAG structure
- Must estimate resources per sub-problem
- Must handle decomposition failures gracefully
- Must support iterative refinement

---

### WS-003: Stage 2 - Manual Review & Override
**Priority**: CRITICAL
**Category**: Workflow Stages
**Source**: Section 3.3 (lines 1988-2258)

**Purpose**: Allow user to review, edit, and approve the AI-generated DecompositionPlan

**Input**:
- ai_generated_plan (DecompositionPlan)

**Process**:
1. Render Manual Review Panel UI
2. Display problem statement and analyzed context
3. Display all sub-problems with dependencies
4. Allow user to:
   - Add/Edit/Delete sub-problems
   - Modify dependencies
   - Change team assignments
   - Change gauntlet assignments
   - Adjust complexity scores
   - Modify evolution parameters
   - Override auto-approval settings
5. Validate user modifications
6. Await user approval
7. Lock approved plan for execution

**UI Requirements**:
- Interactive sub-problem editor
- Dependency visualization (DAG display)
- Team/gauntlet selection dropdowns
- Complexity slider (1-10)
- Evolution mode selector
- Resource estimate display
- Validation error display
- Approve/Reject buttons

**Override Capabilities**:
- Modify any sub-problem description
- Add custom sub-problems
- Delete unwanted sub-problems
- Change solver/patcher teams
- Change red/gold gauntlets
- Adjust evolution parameters
- Modify parallel processing settings
- Override auto-approval criteria

**Validation Requirements**:
- All validations from DM-006 must pass
- No circular dependencies after edits
- All team/gauntlet references must be valid
- Complexity scores must be between 1-10
- At least one sub-problem must remain

**Implementation Requirements**:
- Must pause workflow execution
- Must support interactive UI updates
- Must validate changes in real-time
- Must support save/load of draft plans
- Must support plan templates

---

### WS-004: Stage 3 - Sub-Problem Solving Loop
**Priority**: CRITICAL
**Category**: Workflow Stages
**Source**: Section 3.4 (lines 2259-2298)

**Purpose**: Execute solution generation, critique, and verification loop for each sub-problem

**Input**:
- approved_plan (DecompositionPlan)
- sub_problem (SubProblem)

**Process** (4 steps per sub-problem):

**Step A: Solution Generation (Blue Team)**
1. Invoke assigned solver_team
2. Generate initial solution based on sub-problem description
3. Apply evolution mode if specified (standard/adversarial/quality_diversity)
4. MDAP/MAKER integration if enabled:
   - Decompose into microtasks
   - Apply k-ahead voting
   - Red-flag invalid solutions
5. Store SolutionAttempt

**Step B: Critique (Red Team Gauntlet)**
1. Invoke assigned red_team_gauntlet
2. Execute all rounds of the gauntlet
3. Apply quorum rules per round
4. Generate CritiqueReport
5. Identify flaws and suggest improvements

**Step C: Verification (Gold Team Gauntlet)**
1. Invoke assigned gold_team_gauntlet
2. Execute all rounds of the gauntlet
3. Evaluate across multiple dimensions
4. Generate VerificationReport
5. Check if verification criteria met

**Step D: Iterative Refinement**
1. If critique finds flaws OR verification fails:
   a. Invoke patcher_team
   b. Apply improvements from critique/verification
   c. Generate new SolutionAttempt
   d. Repeat from Step B
2. Stop when:
   - Verification passes AND critique approves
   - Max refinement loops reached
   - User manually intervenes

**Evolution Modes**:
- standard: Generate single solution, verify once
- adversarial: Red team drives iterative improvement
- quality_diversity: Explore multiple quality criteria

**Validation Requirements**:
- Solution must be non-empty
- Critique report must be generated
- Verification report must be generated
- Max refinement loops must not be exceeded
- All gauntlet rounds must complete

**Implementation Requirements**:
- Must support all evolution modes
- Must integrate with MDAP/MAKER when enabled
- Must track refinement loop count
- Must handle solution generation failures
- Must support parallel processing of independent sub-problems

---

### WS-005: Stage 4 - Configurable Reassembly
**Priority**: CRITICAL
**Category**: Workflow Stages
**Source**: Section 3.5 (lines 2299-2322)

**Purpose**: Integrate verified sub-problem solutions into final coherent solution

**Input**:
- verified_sub_solutions (Dict[str, SolutionAttempt])
- approved_plan (DecompositionPlan)

**Process**:
1. Invoke assigned assembler_team
2. Provide all verified sub-problem solutions
3. Provide dependency graph for proper assembly order
4. Generate integrated final solution
5. Handle conflicts between sub-solutions
6. Ensure consistency across assembled solution
7. Apply reassembly strategy if specified
8. Store as final_solution in WorkflowState

**Reassembly Strategies**:
- sequential: Assemble in dependency order
- parallel: Assemble independent components simultaneously
- hierarchical: Assemble in layers based on dependency depth
- conflict_resolution: Detect and resolve conflicts between sub-solutions

**Conflict Resolution**:
- Variable naming conflicts
- Interface mismatches
- Data format inconsistencies
- Logic contradictions
- Resource allocation conflicts

**Validation Requirements**:
- All sub-problem solutions must be verified before assembly
- All dependencies must be satisfied
- Final solution must be non-empty
- Assembly must preserve all verified solution content

**Implementation Requirements**:
- Must support multiple reassembly strategies
- Must detect and handle conflicts
- Must preserve solution quality
- Must track assembly decisions
- Must support manual intervention

---

### WS-006: Stage 5 - Final Verification & Self-Healing
**Priority**: CRITICAL
**Category**: Workflow Stages
**Source**: Section 3.6 (lines 2323-2889)

**Purpose**: Verify final solution and trigger self-healing loop if needed

**Input**:
- final_solution (SolutionAttempt)
- approved_plan (DecompositionPlan)

**Process**:
1. Run final_red_team_gauntlet on final solution
2. Run final_gold_team_gauntlet on final solution
3. Evaluate if both approve:
   - If YES: Mark workflow as completed
   - If NO: Enter self-healing loop

**Self-Healing Loop** (while refinement_loop_count < max_refinement_loops):
1. Parse critique and verification reports
2. Identify specific sub-problems requiring rework
3. For each identified sub-problem:
   a. Reset sub-problem status to "requires_rework"
   b. Invoke patcher_team with feedback
   c. Re-run Stage 3 for affected sub-problems
4. Re-run Stage 4 (reassembly)
5. Re-run final verification
6. If passes: exit loop
7. If fails and loops remain: continue
8. If fails and max loops reached: mark as failed, await user intervention

**Targeted Feedback Parsing**:
- Extract specific sub-problem IDs from feedback
- Map feedback to specific solution components
- Generate specific rework instructions
- Prioritize rework by severity

**Exit Conditions**:
- Both gauntlets approve → SUCCESS
- Max refinement loops exceeded → FAILURE
- User manual intervention → PAUSED
- Unrecoverable error → FAILURE

**Validation Requirements**:
- Must run both final gauntlets
- Must track refinement loop count
- Must not exceed max_refinement_loops
- Must identify specific failing components
- Must support manual override

**Implementation Requirements**:
- Must support automatic rework routing
- Must handle cascading rework (dependent sub-problems)
- Must preserve working solutions
- Must provide clear failure reasons
- Must support manual intervention at any point

---

### WS-007: Stage 6 - Knowledge Extraction & Learning
**Priority**: MEDIUM
**Category**: Workflow Stages
**Source**: Section 3.7 (lines 2890-3618)

**Purpose**: Extract and store knowledge artifacts for future workflow improvements

**Input**:
- completed_workflow_state (WorkflowState)
- all_reports (List[CritiqueReport], List[VerificationReport])

**Process**:
1. Analyze all solutions, critiques, and verifications
2. Extract solution patterns
3. Map problem types to successful approaches
4. Identify critique insights
5. Track team performance metrics
6. Evaluate gauntlet effectiveness
7. Store knowledge artifacts in knowledge base
8. Update performance metrics for teams and gauntlets
9. Generate learning summary

**Artifact Types Extracted**:
- solution_pattern: Successful solution approaches
- problem_solution_mapping: Problem → solution mappings
- critique_insight: Common flaws and detection patterns
- team_performance: Team effectiveness by problem type
- gauntlet_effectiveness: Gauntlet success rates

**Knowledge Base Integration**:
- Vector storage (Qdrant) for semantic search
- Structured storage (SQLite) for metadata
- Caching layer for fast retrieval
- Version tracking for artifacts

**ACE Integration** (when enabled):
- Update skillbook with learned skills
- Store execution feedback for future agent improvement
- Capture successful strategies as reusable skills
- Learn from failures to avoid repetition

**Usage in Future Workflows**:
- Inject relevant solution patterns during decomposition
- Suggest effective teams based on performance history
- Optimize gauntlet parameters based on effectiveness
- Provide exemplars for similar problems

**Validation Requirements**:
- All artifacts must be valid
- Artifacts must reference valid workflow
- Effectiveness scores must be calculated
- Duplicate artifacts must be merged

**Implementation Requirements**:
- Must support incremental learning
- Must support knowledge retrieval
- Must support knowledge versioning
- Must support knowledge cleanup (pruning)
- Must integrate with vector database

---

## PART 3: TEAM & GAUNTLET ARCHITECTURE (43 Requirements)

### TG-001: Team Role Definitions
**Priority**: CRITICAL
**Category**: Team Architecture
**Source**: Section 2.1.1 (lines 1353-1376)

**Blue Team Sub-roles**:
1. **Planners**: Generate initial decomposition strategies
2. **Solvers**: Generate initial solutions for sub-problems
3. **Patchers**: Modify solutions based on critique/verification
4. **Assemblers**: Integrate verified solutions into final product
5. **Optimizers**: Refine solutions for efficiency/performance
6. **Synthesizers**: Combine multiple solution approaches

**Red Team Sub-roles (Assailants)**:
1. **Security Analysts**: Identify security vulnerabilities
2. **Logic Verifiers**: Check logical consistency
3. **Edge Case Explorers**: Test extreme scenarios
4. **Assumption Challengers**: Question underlying assumptions
5. **Compliance Checkers**: Verify standards/regulations adherence

**Gold Team Sub-roles (Judges)**:
1. **Accuracy Judges**: Evaluate factual correctness
2. **Completeness Judges**: Assess requirement coverage
3. **Efficiency Judges**: Measure performance and resource usage
4. **Usability Judges**: Evaluate user-friendliness
5. **Innovation Judges**: Assess novelty and creativity

**Implementation Requirements**:
- Must support all sub-roles
- Must validate sub-role compatibility with team role
- Must track historical performance by sub-role
- Must support dynamic sub-role assignment

---

### TG-002: Team Specialization
**Priority**: HIGH
**Category**: Team Architecture
**Source**: Section 2.1.3 (lines 1381-1392)

**Domain Specialization**:
- Teams can specialize in specific domains (healthcare, finance, software engineering, etc.)
- Domain specialization configured via domain_specialization field
- Used for automatic team selection based on problem domain

**Problem Type Specialization**:
- Teams can specialize in problem types (optimization, prediction, classification)
- Problem type specialization configured via problem_type_specialization field
- Used for automatic team selection based on problem type

**Expertise Mapping**:
- System maintains mapping of team expertise to problem characteristics
- Enables automatic team selection
- Updated based on performance metrics

**Dynamic Team Formation**:
- For complex problems, combine models from different teams
- Form specialized teams based on sub-problem requirements
- Balance workload across available teams

**Validation Requirements**:
- domain_specialization must be from standard domain list
- problem_type_specialization must be from standard problem type list
- Expertise mapping must be kept up-to-date
- Performance metrics must drive specialization updates

---

### TG-003: Gauntlet Programmable Rules
**Priority**: CRITICAL
**Category**: Gauntlet Architecture
**Source**: Section 2.2.1 (lines 1397-1413)

**Flexible Quorums**:
- M out of N agents approval (e.g., 2 of 3)
- Support fractional quorums (e.g., 2/3 majority)
- Per-round quorum configuration

**Per-Agent Requirements**:
- Different minimum score thresholds per agent
- Different criteria per agent in same round
- Override global round rules for specific models

**Multi-Round Logic**:
- Each round has distinct rules
- Round 1: simple majority
- Round 2: unanimity
- Round N: custom criteria

**Per-Agent Approval Counts**:
- Success requires specific agent to achieve certain approvals
- Track approvals across all rounds
- Require minimum approvals from specific agents

**Statistical Thresholds**:
- score_variance to ensure consensus
- Fail if variance too high even if average scores good
- Minimum confidence thresholds

**Collaboration Modes**:
- independent: Judges evaluate without seeing others
- share_previous_feedback: Later rounds see earlier feedback

**Time Constraints**:
- Time limits per round
- Time limits for entire gauntlet
- Timeout handling and escalation

**Resource Constraints**:
- Maximum API calls per round
- Maximum token usage per round
- Resource exhaustion handling

**Implementation Requirements**:
- Must support all rule types
- Must validate rule consistency
- Must track per-agent metrics
- Must handle rule violations gracefully

---

### TG-004: Advanced Gauntlet Configurations
**Priority**: HIGH
**Category**: Gauntlet Architecture
**Source**: Section 2.2.2 (lines 1415-1425)

**Adaptive Gauntlets**:
- Rules adapt based on content being evaluated
- More stringent for complex/critical solutions
- Dynamic threshold adjustment
- Context-aware rule selection

**Hierarchical Gauntlets**:
- Multi-level evaluation tiers
- Increasingly strict criteria per tier
- Tier advancement based on performance
- Early termination on failure

**Competitive Gauntlets**:
- Multiple solutions compete
- Best-performing advance to next round
- Comparison-based evaluation
- Relative scoring

**Collaborative Gauntlets**:
- Models work together to improve solutions
- Iterative refinement within gauntlet
- Consensus building
- Cooperative evaluation

**Cross-Domain Gauntlets**:
- Evaluate from multiple perspectives
- Multi-domain evaluation
- Interdisciplinary criteria
- Cross-domain consistency checks

**Implementation Requirements**:
- Must support all gauntlet types
- Must track adaptation history
- Must handle competitive scenarios
- Must support collaboration protocols

---

### TG-005: Dynamic Gauntlet Adaptation
**Priority**: MEDIUM
**Category**: Gauntlet Architecture
**Source**: Section 2.2.3 (lines 1427-1437)

**Performance-Based Adjustment**:
- Adjust rules based on previous solution performance
- Become more/less stringent as needed
- Learning from historical data
- Automatic tuning

**Feedback-Driven Evolution**:
- Evolve based on user feedback
- Incorporate system performance metrics
- Continuous improvement
- A/B testing of rules

**Contextual Adaptation**:
- Different criteria for different problem types
- Domain-specific rule sets
- Problem complexity awareness
- Context-aware rule selection

**Resource-Aware Adaptation**:
- Adapt resource usage based on availability
- Prioritize critical evaluations when resources limited
- Resource allocation optimization
- Cost-aware rule adjustment

**Implementation Requirements**:
- Must track performance metrics
- Must support feedback incorporation
- Must maintain rule version history
- Must support rollback of changes

---

## PART 4: MDAP INTEGRATION REQUIREMENTS (38 Requirements)

### MDAP-001: Maximal Agentic Decomposition (MAD)
**Priority**: CRITICAL
**Category**: MDAP Core
**Source**: Section 1.5 (lines 101-132)

**Decomposition Requirements**:
- Tasks decomposed into smallest possible subtasks
- Each subtask assigned to focused microagent
- Agent context limited to information sufficient for its step
- Reduces error correlation through context isolation

**Mathematical Formulation**:
- For s-step task with m steps per subtask:
  - Single-agent: a1, ..., as ~ (ψa ◦ M ◦ φ)(x)
  - MAD (m=1): ri+1 ~ M(φ(xi)), ai+1 = ψa(ri+1), xi+1 = ψx(ri+1) ∀i = 0, ..., s-1

**Benefits**:
- Reduces exponential decay of correctness probability
- Limits error propagation
- Enables focused micro-role assignment
- Improves overall reliability

**Implementation Requirements**:
- Must support recursive decomposition
- Must maintain context boundaries
- Must track decomposition depth
- Must validate atomic subtasks

---

### MDAP-002: First-to-Ahead-by-K Voting
**Priority**: CRITICAL
**Category**: MDAP Core
**Source**: Section 1.5 (lines 113-114)

**Voting Algorithm**:
- Sample candidate outputs until one has k more votes than any other
- Statistical approach improves probability of correct solutions
- Follows sequential probability ratio test (SPRT) optimality

**Mathematical Properties**:
- Probability of subtask success: psub = (p^m * k) / (p^m * k + ((1-p) * p^(m-1))^k)
- Probability of full task success: pfull = psub^(s/m)
- Expected cost: Θ((s * kmin) / (v * (2p - 1)))

**Implementation Requirements**:
- Must implement k-ahead voting
- Must track vote counts
- Must detect k-ahead condition
- Must handle voting failures

---

### MDAP-003: Red-Flagging
**Priority**: CRITICAL
**Category**: MDAP Core
**Source**: Section 1.5 (lines 115-116)

**Red-Flag Indicators**:
1. Overly long responses (>750 tokens threshold)
2. Incorrectly formatted responses (format validation)

**Purpose**:
- Detect and discard unreliable outputs
- Reduce correlated errors
- Increase effective success rate
- Improve overall system quality

**Validation**:
- Probability of valid response parsing: v
- Affects overall system cost
- Must balance strictness vs. false positives

**Implementation Requirements**:
- Must implement length checks
- Must implement format validation
- Must configure thresholds
- Must track red-flag rates

---

### MDAP-004: MDAP Orchestrator
**Priority**: CRITICAL
**Category**: MDAP Implementation
**Source**: Section 1.5 (lines 140-150)

**Component**: MDAPOrchestrator (in mdap_engine.py)

**Responsibilities**:
- Execute MDAPTask objects
- Apply k-ahead voting per step
- Implement fallback policies
- Track metrics

**Required Methods**:
- execute_task(task: MDAPTask) -> TaskResult
- apply_voting(candidates: List[Candidate]) -> Winner
- apply_red_flagging(responses: List[Response]) -> List[Response]
- handle_failure(step: MDAPStep) -> FallbackAction

**Implementation Requirements**:
- Must integrate with workflow_engine.py
- Must support configurable k-values
- Must implement red-flagging
- Must handle failures gracefully

---

### MDAP-005: MDAP Task Definition
**Priority**: CRITICAL
**Category**: MDAP Implementation
**Source**: Section 1.5 (lines 144-145)

**Component**: MDAPStep / MDAPTask (in mdap_engine.py)

**MDAPStep Requirements**:
- Define microtask with explicit schema
- Specify prompt for task
- Define schema expectations
- Set task metadata (priority, dependencies)

**MDAPTask Requirements**:
- Collection of MDAPSteps
- Define task-level configuration
- Set global constraints
- Track task progress

**Implementation Requirements**:
- Must support schema validation
- Must support task composition
- Must track step dependencies
- Must maintain task state

---

### MDAP-006: Red-Flagging Engine
**Priority**: HIGH
**Category**: MDAP Implementation
**Source**: Section 1.5 (lines 145-146)

**Component**: RedFlagger / SchemaValidator (in mdap_engine.py)

**RedFlagger Responsibilities**:
- Detect overly long responses
- Detect incorrectly formatted responses
- Apply adaptive thresholds
- Track flagging statistics

**SchemaValidator Responsibilities**:
- Validate JSON schema compliance
- Check required fields
- Validate field types
- Check value ranges

**Implementation Requirements**:
- Must support configurable thresholds
- Must support multiple validation rules
- Must track validation metrics
- Must provide detailed failure reasons

---

### MDAP-007: MDAP Caching
**Priority**: MEDIUM
**Category**: MDAP Implementation
**Source**: Section 1.5 (lines 146-147)

**Component**: MDAPCache (in mdap_engine.py)

**Cache Functionality**:
- Optional TTL cache for validated subtask outputs
- Cache key based on task signature
- Configurable TTL
- Cache size limits

**Cache Invalidation**:
- TTL-based expiration
- LRU eviction when full
- Manual invalidation support
- Cache statistics tracking

**Implementation Requirements**:
- Must support TTL configuration
- Must implement LRU eviction
- Must track cache hit/miss rates
- Must support cache clearing

---

### MDAP-008: Agent Selection
**Priority**: HIGH
**Category**: MDAP Implementation
**Source**: Section 1.5 (lines 147-148)

**Component**: AgentSelector (in mdap_engine.py)

**Selection Criteria**:
- Specialization matching
- Historical performance metrics
- Current workload
- Capability requirements

**Selection Algorithm**:
1. Filter agents by specialization
2. Score by historical performance
3. Adjust for current workload
4. Select best available agent

**Implementation Requirements**:
- Must track agent performance
- Must support workload balancing
- Must handle agent unavailability
- Must provide selection transparency

---

### MDAP-009: MDAP Workflow Integration
**Priority**: CRITICAL
**Category**: MDAP Integration
**Source**: Section 1.5 (lines 160-169)

**Stage 0 Integration** (Content Analysis):
- When WorkflowState.mdap_enabled is true
- Run through mdap_engine.py with explicit JSON schemas
- Fallback to standard analysis if disabled
- Configuration passed via WorkflowState.mdap_config

**Stage 1 Integration** (Decomposition):
- When WorkflowState.mdap_enabled or WorkflowState.maker_enabled is true
- Follow MDAP/MAKER recursive method
- Each step proposes (P1, P2, C) structure
- Produces dependency-aware plan with explicit composition nodes

**Stage 2 Integration** (Manual Review):
- UI surfaces MDAP toggles and config JSON
- See ui_components.py and ui_components_additional.py

**Stage 3 Integration** (Sub-Problem Solving):
- MDAP runs in workflow_engine.py via _generate_solution_with_mdap
- Core orchestration in mdap_engine.py
- Components: MDAPOrchestrator, MDAPStep, MDAPTask, RedFlagger, MDAPCache

**Stage 4/5 Integration** (Reassembly/Verification):
- MDAP outputs feed existing gauntlet evaluation
- Retries and fallback decisions in workflow engine

**Implementation Requirements**:
- Must support enable/disable per workflow
- Must pass configuration through WorkflowState
- Must integrate with existing gauntlet pipeline
- Must maintain backward compatibility

---

### MDAP-010: MDAP Performance Optimization
**Priority**: HIGH
**Category**: MDAP Optimization
**Source**: Section 1.5 (lines 170-363)

**Parallelization Strategy** (lines 175-211):
- Θ(ln s) voting requirement can be parallelized
- Time complexity scales linearly with task length
- Optimal workers: min(available, max(1, int(task_size * 0.1)))
- Optimal batch size: max(1, available_workers // 4)

**Resource Allocation** (lines 213-270):
- Model selection based on task complexity
- Complexity < 3.0: Small models (gpt-3.5-turbo, claude-haiku, llama-3-8b)
- Complexity 3.0-7.0: Medium models (gpt-4, claude-sonnet, llama-3-70b)
- Complexity > 7.0: Large models (gpt-4-turbo, claude-opus, llama-3-405b)
- Resource multiplier: 0.5x, 1.0x, 1.5x respectively

**Caching Mechanisms** (lines 272-363):
- Validated subtask solutions cached and reused
- MDAPCacheManager with max_size and ttl_seconds
- LRU eviction when cache full
- Cache hit rate tracking
- Benefit factor calculation

**Implementation Requirements**:
- Must implement parallelization
- Must optimize resource allocation
- Must implement caching
- Must track performance metrics
- Must support adaptive optimization

---

### MDAP-011: Adaptive Thresholds
**Priority**: MEDIUM
**Category**: MDAP Optimization
**Source**: Section 1.5 (lines 365-451)

**Component**: AdaptiveThresholdManager

**Responsibilities**:
- Update voting threshold based on task performance
- Adjust k based on success rate
- Consider task complexity
- Track performance history

**Adjustment Rules**:
- Success rate < target - 0.05: Increase k
- Success rate > target + 0.05: Decrease k
- Complexity > 7.0: Increase k
- Complexity < 3.0: Decrease k

**Target Success Rate**: 0.95 (95%)

**Implementation Requirements**:
- Must track performance history
- Must maintain recent history (last 100 tasks)
- Must calculate optimal k for specific tasks
- Must respect min_k and max_k bounds

---

### MDAP-012: Load Balancing
**Priority**: MEDIUM
**Category**: MDAP Optimization
**Source**: Section 1.5 (lines 453-534)

**Component**: MDAPLoadBalancer

**Responsibilities**:
- Select optimal agent for given task
- Track agent statistics
- Update agent performance
- Balance workload

**Selection Score Calculation**:
- Load factor (40%): Prefer less loaded agents
- Capability factor (40%): Prefer more capable agents
- Efficiency factor (20%): Prefer faster agents

**Agent Capabilities**:
- gpt-4: critical, complex, analysis, strategy
- gpt-3.5-turbo: routine, simple, fast
- claude-sonnet: analysis, reasoning, creative
- llama-3-70b: technical, coding, complex

**Implementation Requirements**:
- Must track agent request counts
- Must calculate exponential moving averages
- Must update success rates
- Must update response times

---

### MDAP-013: Consistency Checks
**Priority**: MEDIUM
**Category**: MDAP Quality Assurance
**Source**: Section 1.5 (lines 540-599)

**Purpose**: Cross-validate results across multiple agents

**Components**:
- Cross-validation metrics
- Consistency scoring
- Discrepancy resolution
- Semantic validation

**Implementation Requirements**:
- Must calculate pairwise similarities
- Must calculate consistency score
- Must identify discrepancies
- Must generate resolution recommendations

---

### MDAP-014: Convergence Monitoring
**Priority**: MEDIUM
**Category**: MDAP Quality Assurance
**Source**: Section 1.5 (lines 601-741)

**Component**: ConvergenceMonitor

**Tracked Metrics**:
- Convergence rate
- Stagnation detection
- Dynamic threshold adjustment
- Early termination conditions

**Early Termination Conditions**:
- Confidence >= min_confidence
- Achieved k-ahead
- Is stagnant
- Max iterations reached

**Implementation Requirements**:
- Must track voting history
- Must calculate vote entropy
- Must detect stagnation
- Must support early termination

---

### MDAP-015: Error Pattern Analysis
**Priority**: MEDIUM
**Category**: MDAP Quality Assurance
**Source**: Section 1.5 (lines 743-895)

**Component**: ErrorPatternAnalyzer

**Analysis Types**:
- Pattern clustering
- Root cause analysis
- Feedback loops
- Anomaly detection

**Implementation Requirements**:
- Must cluster similar error patterns
- Must identify common failure modes
- Must perform root cause analysis
- Must generate improvement recommendations

---

### MDAP-016: Reliability Metrics
**Priority**: HIGH
**Category**: MDAP Quality Assurance
**Source**: Section 1.5 (lines 897-1115)

**Component**: MDAPReliabilityMetrics

**Tracked Metrics**:
- Success rate tracking
- Consensus quality
- Red-flagging rate analysis
- Agent performance metrics

**Per-Agent Metrics**:
- Total tasks
- Successful tasks
- Average confidence
- Red-flag rate
- Average response time
- Consensus strength average

**Implementation Requirements**:
- Must record metrics for each task
- Must update agent performance
- Must update task type metrics
- Must generate reliability reports
- Must generate system recommendations

---

## PART 5: MAKER FRAMEWORK REQUIREMENTS (27 Requirements)

### MAKER-001: Solution Generation Algorithm
**Priority**: CRITICAL
**Category**: MAKER Core
**Source**: Section 1.6 (lines 1189-1203)

**Algorithm**:
```python
def generate_solution(x0, M, k):
    A = []  # Action list
    x = x0
    for s steps do
        a, x = do_voting(x, M, k)
        Append a to A
    return A
```

**Parameters**:
- x0: Initial state
- M: LLM model
- k: Vote threshold
- s: Number of steps
- A: Action sequence

**Implementation Requirements**:
- Must implement iterative solution generation
- Must maintain state across iterations
- Must track action history
- Must support variable step counts

---

### MAKER-002: Voting Algorithm
**Priority**: CRITICAL
**Category**: MAKER Core
**Source**: Section 1.6 (lines 1205-1216)

**Algorithm**:
```python
def do_voting(x, M, k):
    V = {v: 0 for v in all_possible_votes}
    while True:
        y = get_vote(x, M)
        V[y] = V[y] + 1
        if V[y] >= k + max(V[v] for v in V if v != y):
            return y, next_state(y)
```

**Requirements**:
- Collect votes until k-ahead achieved
- Independent sampling from LLM
- Statistical winner determination
- Next state computation

**Implementation Requirements**:
- Must implement k-ahead detection
- Must track vote counts
- Must handle voting failures
- Must support parallel voting

---

### MAKER-003: Vote Collection
**Priority**: CRITICAL
**Category**: MAKER Core
**Source**: Section 1.6 (lines 1218-1228)

**Algorithm**:
```python
def get_vote(x, M):
    while True:
        r = M(x)
        if not has_red_flags(r):
            return parse_action(r), parse_next_state(r)
```

**Red-Flag Checks**:
- Length check (response not too long)
- Format check (response parseable)
- Content check (no blocked patterns)

**Implementation Requirements**:
- Must implement red-flagging
- Must parse actions
- Must parse next state
- Must handle parsing failures

---

### MAKER-004: Error Correction Scaling
**Priority**: HIGH
**Category**: MAKER Math
**Source**: Section 1.6 (lines 1230-1236)

**Mathematical Properties**:
- Success probability: P_success = (p^k) / (p^k + (1-p)^k)^(s/k)
- Minimum votes: k_min = Θ(ln s)
- Expected cost (m=1): Θ(p^(-1) * s * ln s)

**Scaling Laws**:
- Log-linear scaling with task length
- Exponential improvement with voting
- Linear cost with respect to steps

**Implementation Requirements**:
- Must calculate success probability
- Must determine minimum k
- Must estimate cost
- Must track actual vs. expected

---

### MAKER-005: MAKER Engine
**Priority**: CRITICAL
**Category**: MAKER Implementation
**Source**: Section 1.6 (lines 1283-1290)

**Component**: MakerEngine (in maker_engine.py)

**Responsibilities**:
- Drive step-by-step decision loop for long-horizon tasks
- Manage state transitions
- Coordinate voting
- Handle checkpoints

**Required Methods**:
- execute(task: MakerTask) -> TaskResult
- step(state: MakerState) -> NextAction
- vote(state: MakerState) -> Winner
- checkpoint(state: MakerState) -> None

**Implementation Requirements**:
- Must support long-horizon tasks
- Must maintain state persistence
- Must support recovery from checkpoints
- Must handle failures gracefully

---

### MAKER-006: MAKER Components
**Priority**: CRITICAL
**Category**: MAKER Implementation
**Source**: Section 1.6 (lines 1286-1289)

**MakerStep**: Defines per-step prompts, schema expectations, task metadata

**MakerState**: Captures current state, history, step index

**CheckpointStore**: Persists progress for recovery (default file-backed)

**Implementation Requirements**:
- Must define step schemas
- Must maintain state history
- Must support checkpoint storage
- Must enable recovery

---

### MAKER-007: MAKER Operational Flow
**Priority**: CRITICAL
**Category**: MAKER Implementation
**Source**: Section 1.6 (lines 1292-1300)

**Steps**:
1. Initialize: Build first step prompt from task state
2. Vote: Collect candidates until k-ahead winner emerges
3. Red-Flag: Filter invalid candidates before voting
4. Advance: Apply winning action to produce next state
5. Checkpoint: Persist progress periodically
6. Escalate: If voting stalls, increase k or select higher-capability agents

**Implementation Requirements**:
- Must implement all 6 steps
- Must handle step failures
- Must support escalation
- Must maintain checkpoints

---

### MAKER-008: MAKER Workflow Integration
**Priority**: CRITICAL
**Category**: MAKER Integration
**Source**: Section 1.6 (line 1290)

**Integration Points**:
- workflow_engine.py via _generate_solution_with_maker
- generate_solution_for_sub_problem

**Implementation Requirements**:
- Must integrate with existing workflow
- Must support enable/disable
- Must pass configuration
- Must handle failures

---

## PART 6: ACE INTEGRATION REQUIREMENTS (12 Requirements)

### ACE-001: ACE Core Components
**Priority**: HIGH
**Category**: ACE Core
**Source**: Section 1.7 (lines 1324-1333)

**Components**:
1. **Agent**: Executes tasks using learned skills
2. **Reflector**: Analyzes execution performance (success/failure)
3. **SkillManager**: Updates skillbook with new skills and insights
4. **Skillbook**: Living document of learned strategies (TOON format)

**Implementation Requirements**:
- Must implement all 4 components
- Must support skill learning
- Must maintain skillbook
- Must enable skill retrieval

---

### ACE-002: ACE Integration Points
**Priority**: HIGH
**Category**: ACE Integration
**Source**: Section 1.7 (lines 1334-1339)

**Stage 0 & 1 Integration**:
- Inject learned skills into content analysis prompts
- Inject learned skills into decomposition prompts

**Stage 3 Integration**:
- Inject skills into solution generation prompts
- Capture Red Team and Gold Team feedback
- Update skillbook with feedback

**Stage 5 Integration**:
- Learn from final validation results
- Update skillbook with validation insights

**Implementation Requirements**:
- Must inject skills at all integration points
- Must capture feedback
- Must update skillbook
- Must maintain skill versioning

---

### ACE-003: ACE Benefits
**Priority**: MEDIUM
**Category**: ACE Benefits
**Source**: Section 1.7 (lines 1340-1344)

**Self-Improving Agents**:
- Performance improves 20-35% on complex tasks
- Continuous learning from execution

**Context Preservation**:
- TOON format reduces token usage
- Maintains context while efficient

**Continuous Learning**:
- Failures prevent similar future failures
- Accumulated knowledge over time

**Implementation Requirements**:
- Must track performance improvements
- Must use TOON format
- Must support continuous learning
- Must measure effectiveness

---

## PART 7: QUALITY ASSESSMENT REQUIREMENTS (24 Requirements)

### QA-001: Quality Dimensions
**Priority**: CRITICAL
**Category**: Quality Framework
**Source**: Section 2.1.1 (lines 1370-1376)

**Gold Team Evaluation Dimensions**:
1. **Accuracy**: Factual correctness of solution
2. **Completeness**: Full coverage of requirements
3. **Efficiency**: Performance and resource utilization
4. **Usability**: User-friendliness and accessibility
5. **Innovation**: Novelty and creativity
6. **Security**: Security robustness (if applicable)
7. **Maintainability**: Code quality and maintainability
8. **Scalability**: Ability to scale (if applicable)

**Implementation Requirements**:
- Must support all dimensions
- Must score each dimension (0.0-1.0)
- Must weight dimensions appropriately
- Must provide dimension-specific feedback

---

### QA-002: Critique Categories
**Priority**: CRITICAL
**Category**: Quality Framework
**Source**: Section 2.1.1 (lines 1363-1369)

**Red Team Attack Modes**:
1. **Security Analysis**: Identify vulnerabilities and exploits
2. **Logic Verification**: Check logical consistency
3. **Edge Case Exploration**: Test extreme scenarios
4. **Assumption Challenge**: Question underlying assumptions
5. **Compliance Checking**: Verify standards/regulations adherence

**Implementation Requirements**:
- Must support all attack modes
- Must identify flaws by category
- Must assess flaw severity
- Must provide specific improvement suggestions

---

### QA-003: Success Criteria Definition
**Priority**: HIGH
**Category**: Quality Framework
**Source**: Section 3.1 (lines 1465-1466)

**Requirements**:
- Define 3-7 specific, measurable success criteria
- Criteria must be testable
- Criteria must be unambiguous
- Criteria must cover all requirements

**Implementation Requirements**:
- Must validate criteria count (3-7)
- Must check measurability
- Must verify unambiguity
- Must ensure requirement coverage

---

### QA-004: Consensus Measurement
**Priority**: HIGH
**Category**: Quality Metrics
**Source**: Section 2.2.1 (lines 1407-1408)

**Metrics**:
- score_variance: Statistical variance of judge scores
- consensus_strength: Margin between top scores
- average_score: Mean of all judge scores

**Thresholds**:
- max_score_variance: Fail if variance exceeds threshold
- min_overall_confidence: Minimum average score required

**Implementation Requirements**:
- Must calculate variance
- Must calculate consensus strength
- Must apply thresholds
- Must provide detailed metrics

---

## PART 8: DEPENDENCY ANALYSIS REQUIREMENTS (9 Requirements)

### DA-001: Dependency Graph Validation
**Priority**: CRITICAL
**Category**: Dependency Management
**Source**: Section 5.5 (lines 3753-3754)

**Requirements**:
- Validate that sub-problem dependencies form a DAG
- Detect circular dependencies
- Detect orphan dependencies
- Validate dependency references

**Implementation Requirements**:
- Must implement DAG validation
- Must detect cycles
- Must detect broken references
- Must provide clear error messages

---

### DA-002: Dependency-Based Execution
**Priority**: CRITICAL
**Category**: Dependency Management
**Source**: Section 3.4 (lines 2259-2298)

**Requirements**:
- Execute sub-problems in dependency order
- Parallel execution of independent sub-problems
- Block execution until dependencies satisfied
- Track dependency completion status

**Implementation Requirements**:
- Must implement topological sort
- Must support parallel execution
- Must track completion status
- Must handle dependency failures

---

### DA-003: Cascading Rework
**Priority**: HIGH
**Category**: Dependency Management
**Source**: Section 3.6 (lines 2323-2889)

**Requirements**:
- When sub-problem requires rework, identify dependent sub-problems
- Re-execute dependent sub-problems after rework
- Minimize cascading rework impact
- Track rework propagation

**Implementation Requirements**:
- Must identify dependents
- Must invalidate dependent solutions
- Must trigger re-execution
- Must minimize rework scope

---

## PART 9: RESOURCE ESTIMATION REQUIREMENTS (8 Requirements)

### RE-001: Initial Resource Estimation
**Priority**: HIGH
**Category**: Resource Management
**Source**: Section 3.1 (lines 1484-1490)

**Estimate Components**:
- time_days: Estimated calendar days
- api_tokens: Estimated API token usage
- human_hours: Estimated human oversight hours

**Basis for Estimation**:
- Problem complexity (1-10)
- Domain characteristics
- Technical stack requirements
- Team capabilities

**Implementation Requirements**:
- Must generate initial estimates
- Must update estimates as work progresses
- Must track actual vs. estimated
- Must refine estimation model over time

---

### RE-002: Per-SubProblem Resource Estimation
**Priority**: MEDIUM
**Category**: Resource Management
**Source**: Section 5.5 (lines 3761)

**Requirements**:
- Estimate resources for each sub-problem
- Consider complexity score
- Consider team capabilities
- Consider gauntlet requirements

**Implementation Requirements**:
- Must estimate per sub-problem
- Must aggregate to total estimate
- Must track actual usage
- Must support dynamic adjustment

---

### RE-003: Resource Limits
**Priority**: MEDIUM
**Category**: Resource Management
**Source**: Section 5.6 (lines 3799-3800)

**Requirements**:
- Set resource limits for workflow
- Track resource usage
- Enforce limits
- Alert on approaching limits

**Implementation Requirements**:
- Must enforce limits
- Must track usage in real-time
- Must provide alerts
- Must support manual override

---

## PART 10: CROSS-REFERENCE MATRIX

### Dependency Graph

```
Data Models (Foundation)
├── ModelConfig → Team, GauntletDefinition
├── Team → GauntletDefinition, WorkflowState
├── GauntletRoundRule → GauntletDefinition
├── GauntletDefinition → WorkflowState, SubProblem
├── SubProblem → DecompositionPlan
├── DecompositionPlan → WorkflowState
├── SolutionAttempt → SubProblem, WorkflowState
├── CritiqueReport → SolutionAttempt
├── VerificationReport → SolutionAttempt
├── WorkflowState → All stages
├── KnowledgeArtifact → WorkflowState
└── PerformanceMetrics → Team, GauntletDefinition, WorkflowState

Workflow Stages (Execution)
├── Stage 0 (Content Analysis) → Stage 1
├── Stage 1 (Decomposition) → Stage 2
├── Stage 2 (Manual Review) → Stage 3
├── Stage 3 (Solving Loop) → Stage 4
├── Stage 4 (Reassembly) → Stage 5
├── Stage 5 (Final Verification) → Stage 6 or Loop back to Stage 3
└── Stage 6 (Knowledge Extraction) → End

MDAP Integration (Enhancement)
├── MDAP Engine → Stage 0, 1, 3, 4, 5
├── MAKER Engine → Stage 1, 3
└── ACE Engine → Stage 0, 1, 3, 5

Quality Assurance (Validation)
├── Red Team Gauntlet → Stage 3, 5
├── Gold Team Gauntlet → Stage 3, 5
└── Performance Metrics → All stages
```

### Critical Path Analysis

**Minimum Viable Implementation**:
1. DM-001 through DM-010 (Data Models)
2. WS-001 through WS-006 (Core Workflow Stages)
3. TG-001 through TG-003 (Basic Teams & Gauntlets)
4. QA-001 through QA-004 (Quality Framework)

**Full Production Implementation**:
- All 287 requirements
- Complete MDAP/MAKER/ACE integration
- crewai integration
- Lean 4 integration

---

## PART 11: SUCCESS CRITERIA

### Functional Requirements Success Criteria

**CRITICAL** (Must Have):
- [ ] All 12 data model classes implemented and validated
- [ ] All 6 workflow stages operational
- [ ] Team and gauntlet configuration functional
- [ ] Basic quality assessment working
- [ ] Manual review panel operational
- [ ] Dependency graph validation functional
- [ ] Self-healing loop operational

**HIGH** (Should Have):
- [ ] MDAP integration functional
- [ ] MAKER framework operational
- [ ] Parallel processing of sub-problems
- [ ] Advanced gauntlet configurations
- [ ] Performance metrics tracking
- [ ] Knowledge extraction functional

**MEDIUM** (Nice to Have):
- [ ] ACE integration operational
- [ ] crewai integration functional
- [ ] Lean 4 verification working
- [ ] Adaptive gauntlets
- [ ] Real-time monitoring dashboard
- [ ] Analytics dashboard

### Technical Success Criteria

**Performance**:
- Workflow completes within estimated time
- Resource usage within limits
- API calls optimized with caching
- Parallel processing achieves speedup

**Quality**:
- Red team catches 90%+ of flaws
- Gold team accuracy >95%
- Final solution meets all success criteria
- Self-healing converges in <3 loops

**Reliability**:
- Workflow failure rate <5%
- Recovery from failures automatic
- Data persistence validated
- State consistency maintained

### User Experience Success Criteria

**Usability**:
- Manual review panel intuitive
- Configuration interface clear
- Progress monitoring transparent
- Error messages actionable

**Control**:
- User can override any AI decision
- User can intervene at any stage
- User has microscopic control
- User can configure all parameters

**Transparency**:
- All decisions explained
- All feedback visible
- All metrics available
- All history preserved

---

## APPENDIX A: REQUIREMENTS PRIORITIZATION MATRIX

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Data Models | 10 | 1 | 1 | 0 | 12 |
| Workflow Stages | 7 | 0 | 1 | 0 | 8 |
| Team Architecture | 3 | 2 | 1 | 0 | 6 |
| Gauntlet Architecture | 3 | 3 | 1 | 0 | 7 |
| MDAP Core | 5 | 2 | 1 | 0 | 8 |
| MDAP Implementation | 5 | 4 | 3 | 0 | 12 |
| MDAP Optimization | 0 | 4 | 3 | 0 | 7 |
| MDAP Quality | 0 | 2 | 3 | 0 | 5 |
| MAKER Framework | 5 | 1 | 1 | 0 | 7 |
| ACE Integration | 0 | 2 | 1 | 0 | 3 |
| Quality Assessment | 4 | 2 | 1 | 0 | 7 |
| Dependency Analysis | 2 | 1 | 1 | 0 | 4 |
| Resource Estimation | 1 | 2 | 1 | 0 | 4 |
| crewai Integration | 0 | 8 | 4 | 4 | 16 |
| Lean 4 Integration | 0 | 4 | 2 | 2 | 8 |
| **TOTAL** | **52** | **38** | **25** | **6** | **121** |

**Note**: This is the count of requirement categories. Individual requirement count is 287.

---

## APPENDIX B: IMPLEMENTATION PHASING

### Phase 1: Foundation (Requirements 1-67)
**Data Models & Core Structures**
- DM-001 through DM-012
- Basic validation
- Serialization/deserialization

### Phase 2: Core Workflow (Requirements 68-119)
**Workflow Stages 0-5**
- WS-001 through WS-006
- Basic team/gauntlet support
- Manual review panel

### Phase 3: Quality Framework (Requirements 120-143)
**Quality Assessment**
- QA-001 through QA-004
- Red/Gold team gauntlets
- Consensus measurement

### Phase 4: Advanced Features (Requirements 144-191)
**MDAP Integration**
- MDAP-001 through MDAP-016
- Parallel processing
- Performance optimization

### Phase 5: Enhanced Intelligence (Requirements 192-215)
**MAKER & ACE**
- MAKER-001 through MAKER-008
- ACE-001 through ACE-003
- Learning capabilities

### Phase 6: External Integration (Requirements 216-287)
**crewai & Lean 4**
- crewai integration
- Lean 4 verification
- Advanced orchestration

---

## CONCLUSION

This document provides a comprehensive extraction of all 287 requirements from the Decomposition_Workflow.md specification. The requirements are organized by category, prioritized, and cross-referenced to support systematic implementation.

**Key Takeaways**:
1. **Data Models**: 12 core classes with 67 specific requirements
2. **Workflow**: 6 stages with 52 specific requirements
3. **Teams/Gauntlets**: Flexible, programmable, and adaptive
4. **MDAP/MAKER**: Advanced decomposition and error correction
5. **Quality**: Multi-dimensional assessment with consensus
6. **Integration**: crewai, Lean 4, and ACE for enhanced capabilities

**Next Steps**:
1. Compare this extraction against current implementation
2. Identify gaps and areas needing work
3. Prioritize implementation based on phasing
4. Create detailed implementation tasks
5. Establish validation criteria for each requirement

---

**Document Status**: COMPLETE
**Last Updated**: 2026-01-03
**Prepared By**: Comprehensive Requirements Extraction
