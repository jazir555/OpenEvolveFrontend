import dataclasses
from typing import List, Dict, Any, Optional, Literal, Set
import time

# --- Core Data Structures ---

@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single AI model within a team."""
    model_id: str
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    # Add any other model-specific parameters as needed

@dataclasses.dataclass
class Team:
    """A user-defined group of AI models assigned to a specific role."""
    name: str
    role: Literal["Blue", "Red", "Gold"] # Use Literal for strict type checking
    members: List[ModelConfig]
    description: Optional[str] = None

@dataclasses.dataclass
class GauntletRoundRule:
    """Defines the rules for a single round within a Gauntlet."""
    round_number: int
    # Quorum for the round: M out of N judges must approve
    quorum_required_approvals: int
    quorum_from_panel_size: int # This should typically be len(Team.members)
    
    # Overall confidence threshold for the round
    min_overall_confidence: float = 0.8 # e.g., 0.8 for 80%
    
    # Optional: Statistical thresholds for consensus
    max_score_variance: Optional[float] = None # e.g., 0.1 to ensure judges agree
    
    # Per-judge requirements for this round
    # Example: {"gemini-pro": {"min_score": 0.9, "required_successful_rounds": 3}}
    # The key is the model_id, value is a dict of specific requirements for that model in this round.
    per_judge_requirements: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    
    # Optional: Collaboration mode for judges in this round
    collaboration_mode: Literal["independent", "share_previous_feedback"] = "independent"

@dataclasses.dataclass
class GauntletDefinition:
    """A programmable, multi-round process that a piece of content must pass."""
    name: str
    team_name: str # Name of the Team that runs this Gauntlet
    rounds: List[GauntletRoundRule]
    description: Optional[str] = None
    
    # For Red Team Gauntlets: specific attack modes
    attack_modes: List[str] = dataclasses.field(default_factory=list) # e.g., ["Security Scan", "Edge Case Analysis"]
    
    # For Blue Team Gauntlets: generation mode
    generation_mode: Literal["single_candidate", "multi_candidate_peer_review"] = "single_candidate"

@dataclasses.dataclass
class SubProblem:
    """Represents a single sub-problem in the decomposition plan."""
    id: str
    description: str
    dependencies: List[str] = dataclasses.field(default_factory=list)
    
    # AI suggestions (from Stage 1)
    ai_suggested_evolution_mode: str = "standard"
    ai_suggested_complexity_score: int = 5 # 1-10
    ai_suggested_evaluation_prompt: str = ""
    
    # User-approved configurations (from Stage 2)
    solver_team_name: str = ""
    red_team_gauntlet_name: Optional[str] = None
    gold_team_gauntlet_name: str = ""
    solver_generation_gauntlet_name: Optional[str] = None # Name of the Blue Team Gauntlet for generation
    
    # Specific evolution parameters for this sub-problem (can override global)
    evolution_params: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class DecompositionPlan:
    """The overall plan for decomposing and solving a complex problem."""
    problem_statement: str
    analyzed_context: Dict[str, Any] # Output from Stage 0
    sub_problems: List[SubProblem]
    
    # Global workflow configurations
    max_refinement_loops: int = 3 # For Stage 5 self-healing
    
    # Teams for final stages
    assembler_team_name: str = ""
    final_red_team_gauntlet_name: Optional[str] = None
    final_gold_team_gauntlet_name: str = ""

@dataclasses.dataclass
class SolutionAttempt:
    """Represents a candidate solution for a sub-problem or the final solution."""
    sub_problem_id: str # Or "final_solution" for the main product
    content: str
    generated_by_model: str
    timestamp: float
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list) # To track changes/iterations

@dataclasses.dataclass
class CritiqueReport:
    """Report from a Red Team Gauntlet."""
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool # True if it passed the Red Team (i.e., no critical flaws found)
    reports_by_judge: List[Dict[str, Any]] # Each dict contains model_id, score, justification, targeted_feedback
    summary: str = ""

@dataclasses.dataclass
class VerificationReport:
    """Report from a Gold Team Gauntlet."""
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool # True if it passed the Gold Team
    reports_by_judge: List[Dict[str, Any]] # Each dict contains model_id, score, justification, targeted_feedback
    average_score: float = 0.0
    score_variance: float = 0.0
    summary: str = ""

# --- Workflow State Management ---

@dataclasses.dataclass
class WorkflowState:
    """Manages the state of an active Sovereign-Grade Decomposition Workflow."""
    workflow_id: str
    current_stage: str # e.g., "Content Analysis", "Decomposition", "Sub-Problem Solving"
    current_sub_problem_id: Optional[str] = None
    current_gauntlet_name: Optional[str] = None
    status: str = "running" # "running", "paused", "completed", "failed"
    progress: float = 0.0 # 0.0 to 1.0
    start_time: float = dataclasses.field(default_factory=time.time)
    end_time: Optional[float] = None
    
    decomposition_plan: Optional[DecompositionPlan] = None
    sub_problem_solutions: Dict[str, SolutionAttempt] = dataclasses.field(default_factory=dict)
    final_solution: Optional[SolutionAttempt] = None
    
    solved_sub_problem_ids: Set[str] = dataclasses.field(default_factory=set) # New field
    
    refinement_loop_count: int = 0
    
    # Store reports for auditing
    all_critique_reports: List[CritiqueReport] = dataclasses.field(default_factory=list)
    all_verification_reports: List[VerificationReport] = dataclasses.field(default_factory=list)

    # Store rejected sub-problems with their reports for the Patcher Team
    rejected_sub_problems: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Store the teams and gauntlets used for this specific workflow run
    # This ensures consistency even if global definitions change
    content_analyzer_team: Optional[Team] = None
    planner_team: Optional[Team] = None
    solver_team: Optional[Team] = None
    patcher_team: Optional[Team] = None
    solver_generation_gauntlet: Optional[GauntletDefinition] = None # The GauntletDefinition used by the solver/patcher for generation
    assembler_team: Optional[Team] = None
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None
    final_red_gauntlet: Optional[GauntletDefinition] = None
    final_gold_gauntlet: Optional[GauntletDefinition] = None
    max_refinement_loops: int = 3

