import dataclasses
from typing import List, Dict, Any, Optional, Literal, Set, Set # Import Set for type hinting sets
import time

# --- Core Data Structures ---

@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single AI model within a team.

    Attributes:
        model_id (str): Unique identifier for the AI model (e.g., "gpt-4o", "claude-3-opus").
        api_key (str): API key for authentication with the model provider.
        api_base (str): Base URL for the API endpoint (defaults to OpenAI's).
        temperature (float): Controls randomness in model outputs (0.0-2.0).
        top_p: float = 1.0  # Nucleus sampling parameter (0.0-1.0).
        max_tokens: int = 4096  # Maximum number of tokens to generate.
        frequency_penalty: float = 0.0  # Penalizes new tokens based on their existing frequency.
        presence_penalty: float = 0.0  # Penalizes new tokens based on whether they appear in the text.
        seed: Optional[int] = None  # Seed for reproducible sampling.
        stop_sequences: Optional[List[str]] = None # Up to 4 sequences where the API will stop generating further tokens.
        logprobs: Optional[bool] = None # Whether to return log probabilities of the output tokens or not.
        top_logprobs: Optional[int] = None # An integer between 0 and 5 specifying the number of most likely tokens to return at each token position.
        response_format: Optional[Dict[str, str]] = None # An object specifying the format that the model must output.
        stream: Optional[bool] = None # If set, partial message deltas will be sent, like in ChatGPT.
        user: Optional[str] = None # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    """
    model_id: str
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    n: Optional[int] = 1 # Number of chat completion choices to generate for each input message.
    logit_bias: Optional[Dict[int, int]] = None # Modify the likelihood of specified tokens appearing in the completion.
    stop_sequences: Optional[List[str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    stream: Optional[bool] = None
    user: Optional[str] = None


@dataclasses.dataclass
class Team:
    """A user-defined group of AI models assigned to a specific role within the workflow.

    Attributes:
        name (str): A unique name for the team.
        role (Literal["Blue", "Red", "Gold"]): Specifies the team\'s primary function (e.g., creation, critique, evaluation).
        members (List[ModelConfig]): A list of `ModelConfig` objects defining the AI models that comprise this team.
        description (Optional[str]): An optional human-readable description of the team\'s purpose or specialization.
    """
    name: str
    role: Literal["Blue", "Red", "Gold"]
    members: List[ModelConfig]
    description: Optional[str] = None

@dataclasses.dataclass
class GauntletRoundRule:
    """Defines the rules for a single round within a Gauntlet."""
    round_number: int
    # Quorum for the round: M out of N judges must approve
    quorum_required_approvals: int
    quorum_from_panel_size: int # Total number of team members participating in this round (should typically be len(Team.members))
    
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
    solver_generation_gauntlet_name: Optional[str] = None # Name of the Blue Team Gauntlet used by the solver/patcher for internal generation/peer review
    
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
    workflow_id: str # Unique ID for this workflow run
    workflow_type: Any # The type of workflow being run (e.g., EvolutionWorkflow.SOVEREIGN_DECOMPOSITION). Using Any to avoid circular imports.
    problem_statement: str # The initial problem statement for this workflow
    current_stage: str # e.g., "Content Analysis", "Decomposition", "Sub-Problem Solving"
    current_sub_problem_id: Optional[str] = None
    current_gauntlet_name: Optional[str] = None
    status: str = "running" # Overall status: "running", "paused", "completed", "failed", "awaiting_user_input"
    progress: float = 0.0 # 0.0 to 1.0
    start_time: float = dataclasses.field(default_factory=time.time)
    end_time: Optional[float] = None
    
    decomposition_plan: Optional[DecompositionPlan] = None
    sub_problem_solutions: Dict[str, SolutionAttempt] = dataclasses.field(default_factory=dict)  # Stores verified solutions for each sub-problem
    solved_sub_problem_ids: set[str] = dataclasses.field(default_factory=set) # Stores IDs of sub-problems that have been successfully solved and verified
    rejected_sub_problems: Dict[str, Any] = dataclasses.field(default_factory=dict) # Stores sub-problem IDs that were rejected, along with their critique/verification reports
    final_solution: Optional[SolutionAttempt] = None  # The final assembled solution attempt
    
    refinement_loop_count: int = 0  # Counter for the self-healing loop in Stage 5

    # Store the teams and gauntlets used for this specific workflow run
    # This ensures consistency even if global definitions change
    content_analyzer_team: Optional[Team] = None # The Blue Team responsible for initial content analysis.
    planner_team: Optional[Team] = None # The Blue Team responsible for generating the decomposition plan.
    solver_team: Optional[Team] = None # The Blue Team responsible for generating solutions for sub-problems.
    patcher_team: Optional[Team] = None # The Blue Team responsible for fixing rejected solutions.
    solver_generation_gauntlet: Optional[GauntletDefinition] = None # The GauntletDefinition used by the solver/patcher for internal generation/peer review.
    assembler_team: Optional[Team] = None # The Blue Team responsible for reassembling the final solution.
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None # The Red Team Gauntlet for critiquing sub-problem solutions.
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None # The Gold Team Gauntlet for verifying sub-problem solutions.
    final_red_gauntlet: Optional[GauntletDefinition] = None # The Red Team Gauntlet for critiquing the final assembled solution.
    final_gold_gauntlet: Optional[GauntletDefinition] = None # The Gold Team Gauntlet for verifying the final assembled solution.
    max_refinement_loops: int = 3 # Max iterations for the self-healing loop in Stage 5.

