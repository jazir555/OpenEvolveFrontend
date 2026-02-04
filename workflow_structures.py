import dataclasses
from typing import List, Dict, Any, Optional, Literal, Set, Union
import time
import json
from enum import Enum
from datetime import datetime
import uuid

# ============================================================================
# Lean 4 / LeanAide Integration Data Structures
# ============================================================================

class MathematicalDomain(Enum):
    """
    Enumeration of mathematical domains for classification and verification.

    Attributes:
        ALGEBRA: Abstract algebra, group theory, ring theory, field theory
        ANALYSIS: Real analysis, complex analysis, measure theory
        TOPOLOGY: Point-set topology, algebraic topology
        NUMBER_THEORY: Elementary number theory, algebraic number theory
        COMBINATORICS: Enumerative combinatorics, graph theory
        GEOMETRY: Euclidean geometry, differential geometry
        LOGIC: Mathematical logic, proof theory, model theory
        SET_THEORY: ZFC, axiomatic set theory
        CATEGORY_THEORY: Categories, functors, natural transformations
        LINEAR_ALGEBRA: Vector spaces, matrices, linear transformations
        CALCULUS: Differential and integral calculus
        PROBABILITY: Probability theory, stochastic processes
        GENERAL: General or cross-domain mathematics
    """
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    CATEGORY_THEORY = "category_theory"
    LINEAR_ALGEBRA = "linear_algebra"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    GENERAL = "general"


class VerificationMethod(Enum):
    """
    Enumeration of verification methods available in the system.

    Attributes:
        MANUAL: Manual review by human experts
        AUTOMATED_TESTING: Automated unit and integration tests
        PEER_REVIEW: Peer review by other agents/experts
        LEAN4: Formal verification using Lean 4 theorem prover
        HYBRID: Combination of multiple verification methods
        STATISTICAL: Statistical validation methods
        CROSS_VALIDATION: Cross-validation across multiple models
    """
    MANUAL = "manual"
    AUTOMATED_TESTING = "automated_testing"
    PEER_REVIEW = "peer_review"
    LEAN4 = "lean4"
    Z3 = "z3"
    HYBRID = "hybrid"
    STATISTICAL = "statistical"
    CROSS_VALIDATION = "cross_validation"


class LeanProofStatus(Enum):
    """
    Status of a Lean 4 proof verification.

    Attributes:
        PENDING: Proof is pending verification
        IN_PROGRESS: Proof verification is in progress
        VERIFIED: Proof has been formally verified
        FAILED: Proof verification failed
        PARTIAL: Partial proof with some unproven obligations
        TIMEOUT: Verification timed out
        ERROR: Error during verification
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclasses.dataclass
class LeanProof:
    """
    Represents a Lean 4 formal proof with metadata.

    Attributes:
        proof_id: Unique identifier for this proof
        theorem_name: Name of the theorem being proved
        lean_code: The Lean 4 proof code
        natural_language_statement: Natural language statement of the theorem
        proof_status: Current verification status
        domain: Mathematical domain classification
        complexity_score: Estimated complexity (1-10)
        proof_steps: List of proof step descriptions
        dependencies: List of theorem/lemma dependencies
        verification_time: Time taken for verification in seconds
        elaborated_type: Elaborated Lean type (if available)
        proof_obligations: List of proof obligations that must be satisfied
        tactics_used: List of Lean tactics used in the proof
        metadata: Additional metadata about the proof
        timestamp: When this proof was created
    """
    proof_id: str
    theorem_name: str
    lean_code: str
    natural_language_statement: str
    proof_status: LeanProofStatus = LeanProofStatus.PENDING
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    complexity_score: int = 1
    proof_steps: List[str] = dataclasses.field(default_factory=list)
    dependencies: List[str] = dataclasses.field(default_factory=list)
    verification_time: float = 0.0
    elaborated_type: str = ""
    proof_obligations: List[str] = dataclasses.field(default_factory=list)
    tactics_used: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "proof_id": self.proof_id,
            "theorem_name": self.theorem_name,
            "lean_code": self.lean_code,
            "natural_language_statement": self.natural_language_statement,
            "proof_status": self.proof_status.value,
            "domain": self.domain.value,
            "complexity_score": self.complexity_score,
            "proof_steps": self.proof_steps,
            "dependencies": self.dependencies,
            "verification_time": self.verification_time,
            "elaborated_type": self.elaborated_type,
            "proof_obligations": self.proof_obligations,
            "tactics_used": self.tactics_used,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeanProof":
        """Create LeanProof from dictionary."""
        data = data.copy()
        if isinstance(data.get("proof_status"), str):
            data["proof_status"] = LeanProofStatus(data["proof_status"])
        if isinstance(data.get("domain"), str):
            data["domain"] = MathematicalDomain(data["domain"])
        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate the Lean proof structure and content.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate required fields
        if not self.theorem_name:
            errors.append("theorem_name is required")
        if not self.lean_code:
            errors.append("lean_code is required")

        # Validate Lean code structure
        if self.lean_code:
            # Check for basic Lean structure
            if not any(keyword in self.lean_code for keyword in ["theorem", "lemma", "def"]):
                errors.append("lean_code must contain theorem, lemma, or def")

            # Check for proof structure
            if ":=" not in self.lean_code and "by" not in self.lean_code:
                errors.append("lean_code must contain := or by keyword")

        # Validate complexity score
        if not 1 <= self.complexity_score <= 10:
            errors.append("complexity_score must be between 1 and 10")

        return errors


@dataclasses.dataclass
class LeanTheorem:
    """
    Represents a mathematical theorem with Lean 4 formalization.

    Attributes:
        theorem_id: Unique identifier for this theorem
        name: Name of the theorem
        statement: Natural language statement
        lean_code: Lean 4 formal statement
        domain: Mathematical domain
        keywords: List of relevant keywords
        difficulty: Estimated difficulty (1-10)
        is_verified: Whether the theorem has been verified
        proof: Associated LeanProof (if available)
        related_theorems: List of related theorem IDs
        references: Academic references or citations
        metadata: Additional metadata
    """
    theorem_id: str
    name: str
    statement: str
    lean_code: str
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    keywords: List[str] = dataclasses.field(default_factory=list)
    difficulty: int = 5
    is_verified: bool = False
    proof: Optional[LeanProof] = None
    related_theorems: List[str] = dataclasses.field(default_factory=list)
    references: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "theorem_id": self.theorem_id,
            "name": self.name,
            "statement": self.statement,
            "lean_code": self.lean_code,
            "domain": self.domain.value,
            "keywords": self.keywords,
            "difficulty": self.difficulty,
            "is_verified": self.is_verified,
            "proof": self.proof.to_dict() if self.proof else None,
            "related_theorems": self.related_theorems,
            "references": self.references,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeanTheorem":
        """Create LeanTheorem from dictionary."""
        data = data.copy()
        if isinstance(data.get("domain"), str):
            data["domain"] = MathematicalDomain(data["domain"])
        if data.get("proof") and isinstance(data["proof"], dict):
            data["proof"] = LeanProof.from_dict(data["proof"])
        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate the theorem structure and mathematical content.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate required fields
        if not self.name:
            errors.append("name is required")
        if not self.statement:
            errors.append("statement is required")
        if not self.lean_code:
            errors.append("lean_code is required")

        # Validate difficulty
        if not 1 <= self.difficulty <= 10:
            errors.append("difficulty must be between 1 and 10")

        # Validate Lean code if verified
        if self.is_verified and not self.lean_code.strip():
            errors.append("Verified theorems must have Lean code")

        return errors


@dataclasses.dataclass
class LeanVerificationResult:
    """
    Result of Lean 4 formal verification.

    Attributes:
        verification_id: Unique identifier for this verification
        success: Whether verification succeeded
        theorem_id: ID of the theorem being verified
        proof_id: ID of the proof used for verification
        verification_method: Method used for verification
        status: Detailed status of verification
        confidence_score: Confidence in the verification (0-1)
        verification_time: Time taken for verification
        proof_steps: Steps taken in the proof
        remaining_obligations: List of unproven obligations
        errors: List of errors encountered
        warnings: List of warnings
        server_used: Whether LeanAide server was used
        fallback_used: Whether fallback to simulation was used
        lean_output: Raw output from Lean 4
        metadata: Additional verification metadata
        timestamp: When verification was performed
    """
    verification_id: str
    success: bool
    theorem_id: str
    proof_id: Optional[str] = None
    verification_method: VerificationMethod = VerificationMethod.LEAN4
    status: LeanProofStatus = LeanProofStatus.PENDING
    confidence_score: float = 0.0
    verification_time: float = 0.0
    proof_steps: List[str] = dataclasses.field(default_factory=list)
    remaining_obligations: List[str] = dataclasses.field(default_factory=list)
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    server_used: bool = True
    fallback_used: bool = False
    lean_output: str = ""
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "verification_id": self.verification_id,
            "success": self.success,
            "theorem_id": self.theorem_id,
            "proof_id": self.proof_id,
            "verification_method": self.verification_method.value,
            "status": self.status.value,
            "confidence_score": self.confidence_score,
            "verification_time": self.verification_time,
            "proof_steps": self.proof_steps,
            "remaining_obligations": self.remaining_obligations,
            "errors": self.errors,
            "warnings": self.warnings,
            "server_used": self.server_used,
            "fallback_used": self.fallback_used,
            "lean_output": self.lean_output,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeanVerificationResult":
        """Create LeanVerificationResult from dictionary."""
        data = data.copy()
        if isinstance(data.get("verification_method"), str):
            data["verification_method"] = VerificationMethod(data["verification_method"])
        if isinstance(data.get("status"), str):
            data["status"] = LeanProofStatus(data["status"])
        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate the verification result.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate required fields
        if not self.verification_id:
            errors.append("verification_id is required")
        if not self.theorem_id:
            errors.append("theorem_id is required")

        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            errors.append("confidence_score must be between 0.0 and 1.0")

        # Validate success status consistency
        if self.success and self.status not in [LeanProofStatus.VERIFIED, LeanProofStatus.PARTIAL]:
            errors.append(f"Success=True but status={self.status} is inconsistent")

        return errors


@dataclasses.dataclass
class MathematicalComponent:
    """
    A mathematical component extracted from a problem or solution.

    Attributes:
        component_id: Unique identifier for this component
        type: Type of mathematical component (theorem, lemma, equation, etc.)
        name: Name of the component
        statement: Mathematical statement
        domain: Mathematical domain
        complexity: Estimated complexity (1-10)
        dependencies: List of dependency IDs
        formalized: Whether this has been formalized in Lean
        lean_code: Lean code if formalized
        verification_status: Verification status if applicable
        metadata: Additional metadata
    """
    component_id: str
    type: str  # "theorem", "lemma", "equation", "definition", "conjecture", etc.
    name: str
    statement: str
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    complexity: int = 1
    dependencies: List[str] = dataclasses.field(default_factory=list)
    formalized: bool = False
    lean_code: str = ""
    verification_status: Optional[LeanProofStatus] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_id": self.component_id,
            "type": self.type,
            "name": self.name,
            "statement": self.statement,
            "domain": self.domain.value,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "formalized": self.formalized,
            "lean_code": self.lean_code,
            "verification_status": self.verification_status.value if self.verification_status else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathematicalComponent":
        """Create MathematicalComponent from dictionary."""
        data = data.copy()
        if isinstance(data.get("domain"), str):
            data["domain"] = MathematicalDomain(data["domain"])
        if isinstance(data.get("verification_status"), str):
            data["verification_status"] = LeanProofStatus(data["verification_status"])
        return cls(**data)


# --- Core Data Structures ---

@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single AI model within a team.

    Attributes:
        model_id (str): Unique identifier for the AI model (e.g., "gpt-4o", "claude-3-opus").
        api_key (str): API key for authentication with the model provider.
        api_base (str): Base URL for the API endpoint (defaults to OpenAI's).
        temperature (float): Controls randomness in model outputs (0.0-2.0).
        top_p (float): Nucleus sampling parameter (0.0-1.0).
        max_tokens (int): Maximum number of tokens to generate.
        frequency_penalty (float): Penalizes new tokens based on their existing frequency in the text so far.
        presence_penalty (float): Penalizes new tokens based on whether they appear in the text so far.
        seed (Optional[int]): Seed for reproducible sampling.
        n (Optional[int]): Number of chat completion choices to generate for each input message.
        logit_bias (Optional[Dict[int, int]]): Modify the likelihood of specified tokens appearing in the completion.
        reasoning_effort (Optional[str]): The reasoning effort to apply for the model (e.g., 'low', 'medium', 'high').
        stop_sequences (Optional[List[str]]): Up to 4 sequences where the API will stop generating further tokens.
        logprobs (Optional[bool]): Whether to return log probabilities of the output tokens or not.
        top_logprobs (Optional[int]): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position.
        response_format (Optional[Dict[str, str]]): An object specifying the format that the model must output.
        stream (Optional[bool]): If set, partial message deltas will be sent, like in ChatGPT.
        user (Optional[str]): A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        max_retries (int): Maximum number of retries for API calls.
        timeout (int): Timeout for API calls in seconds.
        organization (Optional[str]): For OpenAI, the organization ID.
        response_model (Optional[str]): For structured output, a Pydantic model or similar (string representation).
        tools (Optional[List[Dict[str, Any]]]): For function calling, a list of tool definitions.
        tool_choice (Optional[Any]): For function calling, control over tool usage (e.g., "auto", "none", {"type": "function", "function": {"name": "my_function"}}).
        system_fingerprint (Optional[str]): For OpenAI, a unique identifier for the model's configuration.
        deployment_id (Optional[str]): For Azure OpenAI, the deployment name.
        encoding_format (Optional[str]): For some models, the encoding format for output (e.g., "base64").
        max_input_tokens (Optional[int]): Maximum number of input tokens.
        stop_token (Optional[str]): A single stop token (alternative to stop_sequences).
        best_of (Optional[int]): Generates best_of completions on the server side and returns the "best".
        logprobs_offset (Optional[int]): Offset for logprobs.
        suffix (Optional[str]): A suffix that will be appended to the end of the generated text.
        presence_penalty_range (Optional[List[float]]): Range for presence penalty.
        frequency_penalty_range (Optional[List[float]]): Range for frequency penalty.
        stop_token_id (Optional[int]): For models that use token IDs for stopping.
        response_json_format (Optional[bool]): If the response should be in JSON format.
        max_output_tokens (Optional[int]): Maximum number of output tokens.
        stream_options (Optional[Dict[str, Any]]): For more granular control over streaming.
        logprobs_type (Optional[str]): To specify the type of log probabilities.
        top_k (Optional[int]): Another common sampling parameter.
        repetition_penalty (Optional[float]): To penalize repeated tokens.
        length_penalty (Optional[float]): To control the length of generated sequences.
        early_stopping (Optional[bool]): For beam search.
        num_beams (Optional[int]): For beam search.
        do_sample (Optional[bool]): To enable/disable sampling.
        temperature_fallback (Optional[float]): A fallback temperature.
        top_p_fallback (Optional[float]): A fallback top_p.
        max_time (Optional[int]): Maximum time to generate a response.
        return_full_text (Optional[bool]): Whether to return the full text or just the generated part.
        tokenizer_config (Optional[Dict[str, Any]]): For tokenizer-specific settings.
        model_kwargs (Optional[Dict[str, Any]]): For any other model-specific keyword arguments.
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
    n: Optional[int] = 1
    logit_bias: Optional[Dict[int, int]] = None
    reasoning_effort: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    stream: Optional[bool] = None
    user: Optional[str] = None
    max_retries: int = 5
    timeout: int = 120
    organization: Optional[str] = None
    response_model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    system_fingerprint: Optional[str] = None
    deployment_id: Optional[str] = None
    encoding_format: Optional[str] = None
    max_input_tokens: Optional[int] = None
    stop_token: Optional[str] = None
    best_of: Optional[int] = None
    logprobs_offset: Optional[int] = None
    suffix: Optional[str] = None
    presence_penalty_range: Optional[List[float]] = None
    frequency_penalty_range: Optional[List[float]] = None
    stop_token_id: Optional[int] = None
    response_json_format: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None
    logprobs_type: Optional[str] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature_fallback: Optional[float] = None
    top_p_fallback: Optional[float] = None
    max_time: Optional[int] = None
    return_full_text: Optional[bool] = None
    tokenizer_config: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    # Domain specialization for the model
    domain_specialization: Optional[List[str]] = None
    # Problem type specialization for the model
    problem_type_specialization: Optional[List[str]] = None
    # Performance metrics for the model
    performance_metrics: Optional[Dict[str, float]] = None
    # Cost per token for the model
    cost_per_token: Optional[float] = None

@dataclasses.dataclass
class Team:
    """
    A user-defined group of AI models assigned to a specific role within the workflow.

    Attributes:
        name (str): A unique name for the team.
        role (Literal["Blue", "Red", "Gold"]): Specifies the team's primary function (e.g., creation, critique, evaluation).
        members (List[ModelConfig]): A list of `ModelConfig` objects defining the AI models that comprise this team.
        description (Optional[str]): An optional human-readable description of the team's purpose or specialization.
        content_analysis_system_prompt (Optional[str]): System prompt for content analysis if this team is used for it.
        content_analysis_user_prompt_template (Optional[str]): User prompt template for content analysis if this team is used for it.
        decomposition_system_prompt (Optional[str]): System prompt for decomposition if this team is used for it.
        decomposition_user_prompt_template (Optional[str]): User prompt template for decomposition if this team is used for it.
        solver_system_prompt (Optional[str]): System prompt for solvers if this team is used for generating solutions.
        solver_user_prompt_template (Optional[str]): User prompt template for solvers if this team is used for generating solutions.
        patcher_system_prompt (Optional[str]): System prompt for patchers if this team is used for fixing rejected solutions.
        patcher_user_prompt_template (Optional[str]): User prompt template for patchers if this team is used for fixing rejected solutions.
        assembler_system_prompt (Optional[str]): System prompt for assemblers if this team is used for reassembling the final solution.
        assembler_user_prompt_template (Optional[str]): User prompt template for assemblers if this team is used for reassembling the final solution.
        red_team_system_prompt (Optional[str]): System prompt for Red Teams when performing critiques.
        red_team_user_prompt_template (Optional[str]): User prompt template for Red Teams when performing critiques.
        gold_team_system_prompt (Optional[str]): System prompt for Gold Teams when performing verifications.
        gold_team_user_prompt_template (Optional[str]): User prompt template for Gold Teams when performing verifications.
    """
    name: str
    role: Literal["Blue", "Red", "Gold"]
    members: List[ModelConfig]
    tenant_id: Optional[str] = None
    description: Optional[str] = None
    content_analysis_system_prompt: Optional[str] = None
    content_analysis_user_prompt_template: Optional[str] = None
    decomposition_system_prompt: Optional[str] = None
    decomposition_user_prompt_template: Optional[str] = None
    solver_system_prompt: Optional[str] = None
    solver_user_prompt_template: Optional[str] = None
    patcher_system_prompt: Optional[str] = None
    patcher_user_prompt_template: Optional[str] = None
    assembler_system_prompt: Optional[str] = None
    assembler_user_prompt_template: Optional[str] = None
    red_team_system_prompt: Optional[str] = None
    red_team_user_prompt_template: Optional[str] = None
    gold_team_system_prompt: Optional[str] = None
    gold_team_user_prompt_template: Optional[str] = None
    # Sub-role for the team (e.g., "Planner", "Solver", "Patcher" for Blue teams)
    sub_role: Optional[str] = None
    # Domain specialization for the team
    domain_specialization: Optional[List[str]] = None
    # Problem type specialization for the team
    problem_type_specialization: Optional[List[str]] = None
    # Performance metrics for the team
    performance_metrics: Optional[Dict[str, float]] = None
    # Team configuration parameters
    team_config: Optional[Dict[str, Any]] = None
    # OpenEvolve metrics for the team
    openevolve_metrics: Optional[List[Dict[str, Any]]] = None
    # NEW: Team type for cost optimization
    team_type: Literal["standard", "swarm", "sovereign"] = "standard"  # Different optimization strategies

@dataclasses.dataclass
class GauntletRoundRule:
    """Defines the specific rules and criteria for a single round within a Gauntlet."""
    round_number: int
    quorum_required_approvals: int
    quorum_from_panel_size: int
    min_overall_confidence: float = 0.0
    max_score_variance: Optional[float] = None
    per_judge_requirements: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    collaboration_mode: Literal["independent", "share_previous_feedback"] = "independent"
    # Time constraints for this round
    time_limit_seconds: Optional[int] = None
    # Resource constraints for this round
    max_api_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    # Adaptive rules for this round
    adaptive_rules: Optional[Dict[str, Any]] = None

    # NEW: Dynamic Voting Configuration
    voting_strategy: Literal["fixed_quorum", "first_to_ahead_by_k"] = "fixed_quorum"

    # If "first_to_ahead_by_k", this defines the margin 'k'
    # Based on paper: k=3 is often sufficient for p=0.99 reliability
    margin_k: Optional[int] = None

    # Maximum votes to cast before forcing a decision (safety valve)
    max_dynamic_votes: Optional[int] = 100

    # Mathematical verification requirements
    required_mathematical_properties: List[str] = dataclasses.field(default_factory=list)  # List of required mathematical properties to verify
    proof_obligation_threshold: float = 0.0  # Minimum proof confidence required
    mathematical_complexity_level: int = 1  # Required verification depth (1-10)
    proof_generation_enabled: bool = False  # Whether to generate formal proofs for this round
    proof_verification_enabled: bool = False  # Whether to verify formal proofs for this round
    mathematical_approach: str = "direct_proof"  # Approach: "direct_proof", "proof_by_contradiction", "inductive", etc.
    verification_timeout: int = 300  # Timeout for mathematical verification in seconds
    proof_storage_enabled: bool = False  # Whether to store generated proofs
    mathematical_quality_threshold: float = 0.0  # Minimum mathematical quality score (0-1)

@dataclasses.dataclass
class GauntletDefinition:
    """A programmable, multi-round process that a piece of content must pass to be approved."""
    name: str
    team_name: str
    rounds: List[GauntletRoundRule]
    tenant_id: Optional[str] = None
    description: Optional[str] = None
    attack_modes: List[str] = dataclasses.field(default_factory=list)
    generation_mode: Literal["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"] = "single_candidate"
    # Gauntlet type
    gauntlet_type: Literal["standard", "adaptive", "hierarchical", "competitive", "collaborative"] = "standard"
    # Performance metrics for the gauntlet
    performance_metrics: Optional[Dict[str, float]] = None
    # Gauntlet configuration parameters
    gauntlet_config: Optional[Dict[str, Any]] = None
    # NEW: Passive Red Flags (Automatic rejection criteria)
    red_flags: Dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "max_token_length": 2000, # Reject if verbosity indicates confusion
        "strict_format_adherence": True, # Reject if JSON repair is needed
        "forbidden_phrases": ["I apologize", "I'm confused", "As an AI"]
    })
    # Lean 4 / LeanAide formal verification configuration
    formal_verification_enabled: bool = False
    verification_methods: List[VerificationMethod] = dataclasses.field(default_factory=lambda: [VerificationMethod.PEER_REVIEW])
    mathematical_requirements: Dict[str, Any] = dataclasses.field(default_factory=dict)
    proof_generation_enabled: bool = False
    automatic_formalization: bool = False
    formal_verification_threshold: float = 0.9
    lean_verification_config: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class SubProblem:
    """Represents a single sub-problem in the decomposition plan, with its own configurations."""
    id: str
    description: str
    dependencies: List[str] = dataclasses.field(default_factory=list)
    ai_suggested_evolution_mode: str = "standard"
    ai_suggested_complexity_score: int = 5
    ai_suggested_evaluation_prompt: str = ""
    content_type: str = "text_general" # New: Content type for the sub-problem's solution
    solver_team_name: str = ""
    red_team_gauntlet_name: Optional[str] = None
    gold_team_gauntlet_name: str = ""
    solver_generation_gauntlet_name: Optional[str] = None
    patcher_team_name: str = ""  # Name of the Blue Team assigned to patch solutions for this sub-problem
    evolution_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # AI suggestions (additional fields from document)
    ai_suggested_team_assignment: Optional[str] = None
    ai_suggested_gauntlet_assignment: Optional[Dict[str, str]] = None
    estimated_resources: Optional[Dict[str, Any]] = None
    potential_approaches: Optional[List[str]] = None
    # Sub-problem status
    status: Literal["pending", "in_progress", "solved", "failed", "requires_rework"] = "pending"
    # Solution attempts for this sub-problem
    solution_attempts: List['SolutionAttempt'] = dataclasses.field(default_factory=list)
    # Performance metrics for this sub-problem
    performance_metrics: Optional[Dict[str, float]] = None
    # OpenEvolve metrics for this sub-problem
    openevolve_metrics: Optional[Dict[str, Any]] = None
    # NEW: Atomic Decomposition Fields
    atomic_mode: bool = False  # If true, decompose recursively into micro-steps
    decomposition_depth: int = 0  # Current depth of decomposition (0 = original, >0 = atomic)
    micro_steps: List['SubProblem'] = dataclasses.field(default_factory=list)  # For atomic mode, contains micro-steps
    # NEW: Context Slicer related fields
    acceptance_criteria: List[str] = dataclasses.field(default_factory=list)  # Criteria for solution acceptance
    solution_requirements: Dict[str, Any] = dataclasses.field(default_factory=dict)  # Requirements for the solution
    specific_constraints: List[str] = dataclasses.field(default_factory=list)  # Constraints specific to this sub-problem
    dependency_outputs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # Outputs from dependencies
    # Sub-problem metadata (including entanglement signals)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Lean 4 / LeanAide mathematical verification fields
    mathematical_components: List[MathematicalComponent] = dataclasses.field(default_factory=list)
    requires_formal_verification: bool = False
    mathematical_domain: Optional[MathematicalDomain] = None
    formal_verification_enabled: bool = False
    mathematical_properties: List[str] = dataclasses.field(default_factory=list)
    lean_theorems: List[LeanTheorem] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class DecompositionPlan:
    """The overall plan for decomposing and solving a complex problem, including global configurations."""
    problem_statement: str
    analyzed_context: Dict[str, Any]
    sub_problems: List[SubProblem]
    max_refinement_loops: int = 3
    auto_approval_enabled: bool = False
    auto_approval_criteria: Optional[Dict[str, Any]] = None
    # MDAP/MAKER configuration
    mdap_enabled: bool = False
    mdap_config: Dict[str, Any] = dataclasses.field(default_factory=dict)
    maker_enabled: bool = False
    maker_config: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Resource limits for the workflow
    resource_limits: Optional[Dict[str, Any]] = None
    # Parallel processing configuration
    parallel_processing_enabled: bool = False
    max_parallel_sub_problems: int = 1
    # Learning configuration
    learning_enabled: bool = False
    learning_config: Optional[Dict[str, Any]] = None
    # Teams and Gauntlets for final stages
    content_analyzer_team_name: str = ""
    planner_team_name: str = ""
    assembler_team_name: str = ""
    final_red_team_gauntlet_name: Optional[str] = None
    final_gold_team_gauntlet_name: str = ""
    # Decomposition metadata (including entanglement matrix)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class SolutionAttempt:
    """Represents a candidate solution for a sub-problem or the final solution at a given point in time."""
    sub_problem_id: str
    content: str
    generated_by_model: str
    timestamp: float
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    # Solution metadata
    solution_type: Optional[str] = None
    solution_approach: Optional[str] = None
    # Solution quality metrics
    quality_metrics: Optional[Dict[str, float]] = None
    # Resource usage for this solution attempt
    resource_usage: Optional[Dict[str, Any]] = None
    # Solution status
    status: Literal["generated", "critiqued", "verified", "rejected", "patched"] = "generated"
    # Related critiques and verifications
    critique_reports: List['CritiqueReport'] = dataclasses.field(default_factory=list)
    verification_reports: List['VerificationReport'] = dataclasses.field(default_factory=list)
    # OpenEvolve metrics for this solution attempt
    openevolve_metrics: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class CritiqueReport:
    """Report generated by a Red Team Gauntlet, detailing identified flaws and overall robustness."""
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool
    reports_by_judge: List[Dict[str, Any]]
    summary: str = ""
    # Critique metadata
    critique_timestamp: float = dataclasses.field(default_factory=time.time)
    # Critique metrics
    overall_score: float = 0.0
    flaw_severity_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Identified flaws
    identified_flaws: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    # Suggested improvements
    suggested_improvements: List[str] = dataclasses.field(default_factory=list)
    # Resource usage for this critique
    resource_usage: Optional[Dict[str, Any]] = None

@dataclasses.dataclass
class VerificationReport:
    """Report generated by a Gold Team Gauntlet, detailing verification results and confidence."""
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool
    reports_by_judge: List[Dict[str, Any]]
    average_score: float = 0.0
    score_variance: float = 0.0
    summary: str = ""
    # Verification metadata
    verification_timestamp: float = dataclasses.field(default_factory=time.time)
    # Verification metrics
    dimension_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Verification criteria
    criteria_met: List[str] = dataclasses.field(default_factory=list)
    criteria_not_met: List[str] = dataclasses.field(default_factory=list)
    # Targeted feedback
    targeted_feedback: Optional[str] = None
    # Resource usage for this verification
    resource_usage: Optional[Dict[str, Any]] = None
    # Lean 4 / LeanAide verification fields
    lean_verification: Optional[LeanVerificationResult] = None
    verification_method: VerificationMethod = VerificationMethod.PEER_REVIEW
    mathematical_verified: bool = False
    formal_proof_available: bool = False
    mathematical_confidence: float = 0.0
    mathematical_components_verified: List[str] = dataclasses.field(default_factory=list)

# --- Knowledge Management ---

# ============================================================================
# Knowledge Artifact Schema (Phase 1 Implementation)
# ============================================================================

@dataclasses.dataclass
class KnowledgeArtifact:
    """
    Represents a piece of knowledge extracted from a workflow execution.
    
    This is the base class for all knowledge artifacts in the system.
    It captures metadata about when, where, and how the knowledge was generated,
    along with content and usage tracking.
    
    Attributes:
        artifact_id: Unique identifier (UUID-based) for this artifact
        artifact_type: Type of knowledge artifact (solution_pattern, team_performance, etc.)
        source_workflow_id: ID of the workflow this artifact was extracted from
        source_stage: Workflow stage (0-6) where this artifact was created
        timestamp: When this artifact was created
        confidence: Confidence score (0.0-1.0) in the artifact's validity
        title: Human-readable title for the artifact
        description: Detailed description of the artifact's content
        content: Structured content specific to the artifact type
        metadata: Additional metadata about the artifact
        related_artifacts: List of related artifact IDs for linking
        citations: List of citations or references
        tags: List of searchable tags
        usage_count: Number of times this artifact has been used
        last_used: Timestamp of last usage (if any)
        effectiveness_score: Measured effectiveness score (if evaluated)
    """
    artifact_id: str
    artifact_type: Literal[
        "solution_pattern", 
        "team_performance", 
        "gauntlet_effectiveness", 
        "critique_insight", 
        "decomposition_strategy", 
        "verification_method",
        "adr",
        "refinement_template"
    ]
    source_workflow_id: str
    source_stage: Literal[0, 1, 2, 3, 4, 5, 6]
    timestamp: datetime
    confidence: float
    title: str
    description: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    related_artifacts: List[str] = dataclasses.field(default_factory=list)
    citations: List[str] = dataclasses.field(default_factory=list)
    tags: List[str] = dataclasses.field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    effectiveness_score: Optional[float] = None

    def __post_init__(self):
        """Generate artifact_id if not provided."""
        if not self.artifact_id:
            self.artifact_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()

    @property
    def id(self) -> str:
        """Backward-compatible alias for artifact_id."""
        return self.artifact_id

    def validate(self) -> bool:
        """
        Validates all fields of the knowledge artifact.
        
        Raises:
            ValueError: If any field is invalid
            
        Returns:
            True if all fields are valid
        """
        # Validate artifact_id
        if not self.artifact_id or not isinstance(self.artifact_id, str):
            raise ValueError("artifact_id must be a non-empty string")
        
        # Validate artifact_type
        valid_types = [
            "solution_pattern", "team_performance", "gauntlet_effectiveness",
            "critique_insight", "decomposition_strategy", "verification_method",
            "adr", "refinement_template"
        ]
        if self.artifact_type not in valid_types:
            raise ValueError(f"artifact_type must be one of {valid_types}")
        
        # Validate source_workflow_id
        if not self.source_workflow_id or not isinstance(self.source_workflow_id, str):
            raise ValueError("source_workflow_id must be a non-empty string")
        
        # Validate source_stage
        if self.source_stage not in [0, 1, 2, 3, 4, 5, 6]:
            raise ValueError("source_stage must be an integer between 0 and 6")
        
        # Validate timestamp
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        # Validate confidence
        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a float between 0.0 and 1.0")
        
        # Validate title
        if not self.title or not isinstance(self.title, str):
            raise ValueError("title must be a non-empty string")
        
        # Validate description
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        
        # Validate content
        if not isinstance(self.content, dict):
            raise ValueError("content must be a dictionary")
        
        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        
        # Validate related_artifacts (should be list of strings)
        if not isinstance(self.related_artifacts, list):
            raise ValueError("related_artifacts must be a list")
        if not all(isinstance(aid, str) for aid in self.related_artifacts):
            raise ValueError("all related_artifacts must be strings")
        
        # Validate citations
        if not isinstance(self.citations, list):
            raise ValueError("citations must be a list")
        if not all(isinstance(c, str) for c in self.citations):
            raise ValueError("all citations must be strings")
        
        # Validate tags
        if not isinstance(self.tags, list):
            raise ValueError("tags must be a list")
        if not all(isinstance(t, str) for t in self.tags):
            raise ValueError("all tags must be strings")
        
        # Validate usage_count
        if not isinstance(self.usage_count, int) or self.usage_count < 0:
            raise ValueError("usage_count must be a non-negative integer")
        
        # Validate last_used (optional, but if present must be datetime)
        if self.last_used is not None and not isinstance(self.last_used, datetime):
            raise ValueError("last_used must be a datetime object or None")
        
        # Validate effectiveness_score (optional, but if present must be float 0-1)
        if self.effectiveness_score is not None:
            if not isinstance(self.effectiveness_score, (int, float)):
                raise ValueError("effectiveness_score must be a float or None")
            if not 0.0 <= self.effectiveness_score <= 1.0:
                raise ValueError("effectiveness_score must be between 0.0 and 1.0")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the artifact to a dictionary with ISO datetime format.
        
        Returns:
            Dictionary representation of the artifact
        """
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "source_workflow_id": self.source_workflow_id,
            "source_stage": self.source_stage,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "metadata": self.metadata,
            "related_artifacts": self.related_artifacts,
            "citations": self.citations,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "effectiveness_score": self.effectiveness_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        """
        Deserializes an artifact from a dictionary.
        
        Args:
            data: Dictionary containing artifact data
            
        Returns:
            KnowledgeArtifact instance
        """
        data = data.copy()
        
        # Parse timestamp from ISO format
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Parse last_used from ISO format
        if data.get("last_used") and isinstance(data["last_used"], str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        
        return cls(**data)

    def record_usage(self):
        """Records that this artifact has been used."""
        self.usage_count += 1
        self.last_used = datetime.now()

    def add_related_artifact(self, artifact_id: str):
        """Adds a related artifact ID if not already present."""
        if artifact_id not in self.related_artifacts:
            self.related_artifacts.append(artifact_id)

    def add_tag(self, tag: str):
        """Adds a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_citation(self, citation: str):
        """Adds a citation if not already present."""
        if citation not in self.citations:
            self.citations.append(citation)


@dataclasses.dataclass
class SolutionPatternArtifact(KnowledgeArtifact):
    """
    Specialized artifact for solution patterns extracted from successful workflows.
    
    This artifact captures reusable solution approaches that can be applied
    to similar problems in the future.
    
    Attributes:
        pattern_category: Category classification for the pattern
        problem_domains: List of problem domains this pattern applies to
        approach_signature: Signature defining the approach (parameters, constraints)
        success_rate: Historical success rate (0.0-1.0) of this pattern
        avg_execution_time: Average execution time in seconds
    """
    pattern_category: str = ""
    problem_domains: List[str] = dataclasses.field(default_factory=list)
    approach_signature: Dict[str, Any] = dataclasses.field(default_factory=dict)
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    artifact_type: str = dataclasses.field(default="solution_pattern", init=False)

    def validate(self) -> bool:
        """Validates solution pattern specific fields."""
        # First validate base fields
        super().validate()
        
        # Validate pattern_category
        if not isinstance(self.pattern_category, str):
            raise ValueError("pattern_category must be a string")
        
        # Validate problem_domains
        if not isinstance(self.problem_domains, list):
            raise ValueError("problem_domains must be a list")
        if not all(isinstance(d, str) for d in self.problem_domains):
            raise ValueError("all problem_domains must be strings")
        
        # Validate approach_signature
        if not isinstance(self.approach_signature, dict):
            raise ValueError("approach_signature must be a dictionary")
        
        # Validate success_rate
        if not isinstance(self.success_rate, (int, float)):
            raise ValueError("success_rate must be a float")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be between 0.0 and 1.0")
        
        # Validate avg_execution_time
        if not isinstance(self.avg_execution_time, (int, float)):
            raise ValueError("avg_execution_time must be a number")
        if self.avg_execution_time < 0:
            raise ValueError("avg_execution_time must be non-negative")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dictionary including solution pattern fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "pattern_category": self.pattern_category,
            "problem_domains": self.problem_domains,
            "approach_signature": self.approach_signature,
            "success_rate": self.success_rate,
            "avg_execution_time": self.avg_execution_time,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolutionPatternArtifact":
        """Deserializes from dictionary."""
        data = data.copy()
        # Remove artifact_type as it's set automatically
        data.pop('artifact_type', None)
        # Parse timestamp fields
        if data.get('timestamp') and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_used') and isinstance(data['last_used'], str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclasses.dataclass
class TeamPerformanceArtifact(KnowledgeArtifact):
    """
    Specialized artifact for team performance data.
    
    Captures metrics and insights about team effectiveness,
    composition, and optimal use cases.
    
    Attributes:
        team_id: Identifier for the team being evaluated
        team_composition: Details about team members and configuration
        performance_metrics: Quantitative performance metrics
        strengths: Identified team strengths
        weaknesses: Identified team weaknesses
        optimal_problem_types: Problem types this team excels at
    """
    team_id: str = ""
    team_composition: Dict[str, Any] = dataclasses.field(default_factory=dict)
    performance_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    strengths: List[str] = dataclasses.field(default_factory=list)
    weaknesses: List[str] = dataclasses.field(default_factory=list)
    optimal_problem_types: List[str] = dataclasses.field(default_factory=list)
    artifact_type: str = dataclasses.field(default="team_performance", init=False)

    def validate(self) -> bool:
        """Validates team performance specific fields."""
        # First validate base fields
        super().validate()
        
        # Validate team_id
        if not isinstance(self.team_id, str):
            raise ValueError("team_id must be a string")
        
        # Validate team_composition
        if not isinstance(self.team_composition, dict):
            raise ValueError("team_composition must be a dictionary")
        
        # Validate performance_metrics
        if not isinstance(self.performance_metrics, dict):
            raise ValueError("performance_metrics must be a dictionary")
        for key, value in self.performance_metrics.items():
            if not isinstance(key, str):
                raise ValueError("all performance_metrics keys must be strings")
            if not isinstance(value, (int, float)):
                raise ValueError(f"performance_metrics['{key}'] must be a number")
        
        # Validate strengths
        if not isinstance(self.strengths, list):
            raise ValueError("strengths must be a list")
        if not all(isinstance(s, str) for s in self.strengths):
            raise ValueError("all strengths must be strings")
        
        # Validate weaknesses
        if not isinstance(self.weaknesses, list):
            raise ValueError("weaknesses must be a list")
        if not all(isinstance(w, str) for w in self.weaknesses):
            raise ValueError("all weaknesses must be strings")
        
        # Validate optimal_problem_types
        if not isinstance(self.optimal_problem_types, list):
            raise ValueError("optimal_problem_types must be a list")
        if not all(isinstance(pt, str) for pt in self.optimal_problem_types):
            raise ValueError("all optimal_problem_types must be strings")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dictionary including team performance fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "team_id": self.team_id,
            "team_composition": self.team_composition,
            "performance_metrics": self.performance_metrics,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "optimal_problem_types": self.optimal_problem_types,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamPerformanceArtifact":
        """Deserializes from dictionary."""
        data = data.copy()
        data.pop('artifact_type', None)
        if data.get('timestamp') and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_used') and isinstance(data['last_used'], str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclasses.dataclass
class GauntletEffectivenessArtifact(KnowledgeArtifact):
    """
    Specialized artifact for gauntlet effectiveness data.
    
    Captures metrics about how well a gauntlet (evaluation process)
    identifies issues and approves quality solutions.
    
    Attributes:
        gauntlet_id: Identifier for the gauntlet being evaluated
        rule_effectiveness: Effectiveness scores for individual rules
        catch_rate: Rate at which the gauntlet catches issues (0.0-1.0)
        false_positive_rate: Rate of false positives (0.0-1.0)
        optimal_contexts: Contexts where this gauntlet performs best
    """
    gauntlet_id: str = ""
    rule_effectiveness: Dict[str, float] = dataclasses.field(default_factory=dict)
    catch_rate: float = 0.0
    false_positive_rate: float = 0.0
    optimal_contexts: List[str] = dataclasses.field(default_factory=list)
    artifact_type: str = dataclasses.field(default="gauntlet_effectiveness", init=False)

    def validate(self) -> bool:
        """Validates gauntlet effectiveness specific fields."""
        # First validate base fields
        super().validate()
        
        # Validate gauntlet_id
        if not isinstance(self.gauntlet_id, str):
            raise ValueError("gauntlet_id must be a string")
        
        # Validate rule_effectiveness
        if not isinstance(self.rule_effectiveness, dict):
            raise ValueError("rule_effectiveness must be a dictionary")
        for key, value in self.rule_effectiveness.items():
            if not isinstance(key, str):
                raise ValueError("all rule_effectiveness keys must be strings")
            if not isinstance(value, (int, float)):
                raise ValueError(f"rule_effectiveness['{key}'] must be a number")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rule_effectiveness['{key}'] must be between 0.0 and 1.0")
        
        # Validate catch_rate
        if not isinstance(self.catch_rate, (int, float)):
            raise ValueError("catch_rate must be a float")
        if not 0.0 <= self.catch_rate <= 1.0:
            raise ValueError("catch_rate must be between 0.0 and 1.0")
        
        # Validate false_positive_rate
        if not isinstance(self.false_positive_rate, (int, float)):
            raise ValueError("false_positive_rate must be a float")
        if not 0.0 <= self.false_positive_rate <= 1.0:
            raise ValueError("false_positive_rate must be between 0.0 and 1.0")
        
        # Validate optimal_contexts
        if not isinstance(self.optimal_contexts, list):
            raise ValueError("optimal_contexts must be a list")
        if not all(isinstance(c, str) for c in self.optimal_contexts):
            raise ValueError("all optimal_contexts must be strings")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dictionary including gauntlet effectiveness fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "gauntlet_id": self.gauntlet_id,
            "rule_effectiveness": self.rule_effectiveness,
            "catch_rate": self.catch_rate,
            "false_positive_rate": self.false_positive_rate,
            "optimal_contexts": self.optimal_contexts,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GauntletEffectivenessArtifact":
        """Deserializes from dictionary."""
        data = data.copy()
        data.pop('artifact_type', None)
        if data.get('timestamp') and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_used') and isinstance(data['last_used'], str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclasses.dataclass
class CritiqueInsightArtifact(KnowledgeArtifact):
    """
    Specialized artifact for critique insights.
    
    Captures patterns and learnings from critique processes,
    including common issues and improvement suggestions.
    
    Attributes:
        critique_type: Type of critique (code_review, solution_review, etc.)
        common_issues: Frequently identified issues
        improvement_suggestions: Suggested improvements
    """
    critique_type: str = ""
    common_issues: List[str] = dataclasses.field(default_factory=list)
    improvement_suggestions: List[str] = dataclasses.field(default_factory=list)
    artifact_type: str = dataclasses.field(default="critique_insight", init=False)

    def validate(self) -> bool:
        """Validates critique insight specific fields."""
        # First validate base fields
        super().validate()
        
        # Validate critique_type
        if not isinstance(self.critique_type, str):
            raise ValueError("critique_type must be a string")
        
        # Validate common_issues
        if not isinstance(self.common_issues, list):
            raise ValueError("common_issues must be a list")
        if not all(isinstance(i, str) for i in self.common_issues):
            raise ValueError("all common_issues must be strings")
        
        # Validate improvement_suggestions
        if not isinstance(self.improvement_suggestions, list):
            raise ValueError("improvement_suggestions must be a list")
        if not all(isinstance(s, str) for s in self.improvement_suggestions):
            raise ValueError("all improvement_suggestions must be strings")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dictionary including critique insight fields."""
        base_dict = super().to_dict()
        base_dict.update({
            "critique_type": self.critique_type,
            "common_issues": self.common_issues,
            "improvement_suggestions": self.improvement_suggestions,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CritiqueInsightArtifact":
        """Deserializes from dictionary."""
        data = data.copy()
        data.pop('artifact_type', None)
        if data.get('timestamp') and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_used') and isinstance(data['last_used'], str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


# Legacy KnowledgeArtifact for backward compatibility
# This is kept for existing code that uses the old structure
@dataclasses.dataclass
class LegacyKnowledgeArtifact:
    """Represents a piece of knowledge extracted from a workflow execution (Legacy version)."""
    id: str  # Unique identifier for this knowledge artifact
    artifact_type: Literal["solution_pattern", "problem_solution_mapping", "critique_insight", "team_performance", "gauntlet_effectiveness"]
    content: Dict[str, Any]  # Content of the knowledge artifact
    source_workflow_id: str  # ID of the workflow this artifact was extracted from
    extraction_timestamp: float = dataclasses.field(default_factory=time.time)
    # Artifact metadata
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    # Artifact usage metrics
    usage_count: int = 0
    effectiveness_score: float = 0.0
    # Artifact relationships
    related_artifacts: List[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class PerformanceMetrics:
    """Represents performance metrics for a team, gauntlet, or workflow."""
    entity_type: Literal["team", "gauntlet", "workflow"]
    entity_id: str
    metrics: Dict[str, float]
    timestamp: float = dataclasses.field(default_factory=time.time)
    # Metrics metadata
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    # Metrics context
    context: Optional[Dict[str, Any]] = None

# --- Workflow State Management ---

@dataclasses.dataclass
class WorkflowState:
    """Manages the state of an active Sovereign-Grade Decomposition Workflow."""
    workflow_id: str
    workflow_type: Any
    problem_statement: str
    current_stage: str
    tenant_id: Optional[str] = None
    current_sub_problem_id: Optional[str] = None
    current_gauntlet_name: Optional[str] = None
    status: str = "running"
    progress: float = 0.0
    start_time: float = dataclasses.field(default_factory=time.time)
    end_time: Optional[float] = None
    decomposition_plan: Optional[DecompositionPlan] = None
    sub_problem_solutions: Dict[str, SolutionAttempt] = dataclasses.field(default_factory=dict)
    solved_sub_problem_ids: Set[str] = dataclasses.field(default_factory=set)
    rejected_sub_problems: Dict[str, Any] = dataclasses.field(default_factory=dict)
    final_solution: Optional[SolutionAttempt] = None
    refinement_loop_count: int = 0
    # Auto-refine toggle for analytics-driven refinement
    auto_refine_enabled: bool = False
    # Fractal entanglement matrix for dependency propagation
    entanglement_matrix: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)
    entanglement_strict_mode: bool = False
    
    # Store the specific teams and gauntlets used for THIS workflow run.
    # This ensures consistency even if global definitions in TeamManager/GauntletManager change later.
    content_analyzer_team: Optional[Team] = None
    planner_team: Optional[Team] = None
    solver_team: Optional[Team] = None
    patcher_team: Optional[Team] = None
    assembler_team: Optional[Team] = None
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None
    solver_generation_gauntlet: Optional[GauntletDefinition] = None
    final_red_gauntlet: Optional[GauntletDefinition] = None
    final_gold_gauntlet: Optional[GauntletDefinition] = None
    max_refinement_loops: int = 3 # Max iterations for the self-healing loop
    all_critique_reports: List[CritiqueReport] = dataclasses.field(default_factory=list)
    all_verification_reports: List[VerificationReport] = dataclasses.field(default_factory=list)
    # Resource usage for the workflow
    resource_usage: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Performance metrics for the workflow
    performance_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Knowledge artifacts extracted from the workflow
    knowledge_artifacts: List[KnowledgeArtifact] = dataclasses.field(default_factory=list)
    # OpenEvolve metrics for the workflow
    openevolve_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # MDAP/MAKER configuration overrides for this workflow run
    mdap_enabled: bool = False
    mdap_config: Dict[str, Any] = dataclasses.field(default_factory=dict)
    maker_enabled: bool = False
    maker_config: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # OpenEvolve Parameters (User-configurable via UI) - Complete set
    max_iterations: int = 100
    population_size: int = 50
    num_islands: int = 5
    migration_interval: int = 10
    migration_rate: float = 0.1
    archive_size: int = 10
    elite_ratio: float = 0.1
    exploration_ratio: float = 0.7
    exploitation_ratio: float = 0.3
    checkpoint_interval: int = 10
    feature_dimensions: List[str] = dataclasses.field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    diversity_metric: str = "edit_distance"
    enable_artifacts: bool = True
    cascade_evaluation: bool = False
    cascade_thresholds: List[float] = dataclasses.field(default_factory=lambda: [0.5, 0.75, 0.9])
    use_llm_feedback: bool = True
    llm_feedback_weight: float = 0.5
    parallel_evaluations: int = 4
    distributed: bool = False
    template_dir: str = "./templates"
    num_top_programs: int = 5
    num_diverse_programs: int = 5
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    use_meta_prompting: bool = True
    meta_prompt_weight: float = 0.5
    include_artifacts: bool = True
    max_artifact_bytes: int = 1048576 # 1 MB
    artifact_security_filter: bool = True
    early_stopping_patience: int = 10
    convergence_threshold: float = 0.01
    early_stopping_metric: str = "fitness"
    memory_limit_mb: int = 2048
    cpu_limit: float = 1.0
    random_seed: Optional[int] = None
    db_path: str = "./openevolve.db"
    in_memory: bool = False
    diff_based_evolution: bool = False
    max_code_length: int = 10000
    evolution_trace_enabled: bool = False
    evolution_trace_format: str = "json"
    evolution_trace_include_code: bool = False
    evolution_trace_include_prompts: bool = False
    evolution_trace_output_path: str = "./evolution_trace"
    evolution_trace_buffer_size: int = 100
    evolution_trace_compress: bool = False
    log_level: str = "INFO"

    # Complete set of ALL 272+ OpenEvolve parameters from UI (organized by category)
    # This stores the full configuration from parameter_definitions.py
    openevolve_parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)
    log_dir: str = "./logs"
    api_timeout: int = 60
    api_retries: int = 3
    api_retry_delay: float = 1.0
    artifact_size_threshold: int = 5242880 # 5 MB
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 7
    diversity_reference_size: int = 100
    max_retries_eval: int = 3
    evaluator_timeout: int = 30
    evaluator_models: Optional[List[Dict[str, Any]]] = None
    
    # Advanced OpenEvolve research features
    double_selection: bool = False
    adaptive_feature_dimensions: bool = False
    test_time_compute: bool = False
    optillm_integration: bool = False
    plugin_system: bool = False
    hardware_optimization: bool = False
    multi_strategy_sampling: bool = False
    ring_topology: bool = False
    controlled_gene_flow: bool = False
    auto_diff: bool = False
    symbolic_execution: bool = False
    coevolutionary_approach: bool = False
    
    # Additional LLM parameters for comprehensive control
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    reasoning_effort: Optional[str] = None
    system_message: str = ""
    evaluator_system_message: str = ""
    
    # Quality Diversity specific parameters
    qd_algorithm: str = "map_elites"  # map_elites, cvt_map_elites, novelty_search
    qd_selection_pressure: float = 0.8
    qd_mutation_rate: float = 0.1
    qd_crossover_rate: float = 0.7
    qd_novelty_threshold: float = 0.1
    qd_archive_threshold: float = 0.5
    
    # Multi-objective specific parameters
    mo_algorithm: str = "nsga2"  # nsga2, spea2, moea_d
    mo_crossover_prob: float = 0.9
    mo_mutation_prob: float = 0.1
    mo_tournament_size: int = 3
    mo_reference_point: Optional[List[float]] = None
    
    # Adversarial specific parameters
    adversarial_rounds: int = 5
    adversarial_attack_budget: int = 10
    adversarial_defense_budget: int = 10
    adversarial_success_threshold: float = 0.8
    
    # Prompt optimization specific parameters
    prompt_max_length: int = 2000
    prompt_min_length: int = 50
    prompt_optimization_target: str = "performance"  # performance, brevity, clarity
    prompt_evaluation_samples: int = 10
    
    # Code evolution specific parameters
    code_language: str = "python"
    code_style_guide: str = "pep8"
    code_complexity_limit: int = 15
    code_test_coverage_target: float = 0.8
    code_security_scan: bool = True
    
    # Document evolution specific parameters
    document_max_words: int = 5000
    document_min_words: int = 100
    document_readability_target: str = "college"  # elementary, middle, high_school, college, graduate
    document_tone: str = "professional"  # casual, professional, academic, technical
    document_format: str = "markdown"  # markdown, html, latex, plain_text

    # LeanAide Formal Verification Parameters
    leanaide_enabled: bool = False  # Enable LeanAide formal verification for mathematical problems
    leanaide_host: str = "localhost"  # LeanAide server host
    leanaide_port: int = 7654  # LeanAide server port
    leanaide_confidence_threshold: float = 0.7  # Minimum confidence for formal verification success
    leanaide_auto_detect_math: bool = True  # Automatically detect mathematical problems
    leanaide_require_formal_proof: bool = False  # Require formal proof generation
    leanaide_store_proofs: bool = True  # Store generated proofs
    leanaide_verification_method: Literal["leanaide_only", "leanaide_primary", "standard_primary"] = "standard_primary"  # Verification method priority
    leanaide_timeout: int = 300  # Timeout for LeanAide verification in seconds

    # CrewAI integration attributes
    CrewAI_workflow_id: Optional[str] = None
    id_to_ticket_id_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    ticket_id_to_subproblem_id_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    
    def get_crewai_integration(self, api_base: str, api_key: str, project_id: str):
        """
        Get a configured CrewAI integration manager for this workflow
        
        Args:
            api_base: Base URL for CrewAI API
            api_key: API key for authentication
            project_id: Project ID in CrewAI
            
        Returns:
            CrewAIIntegrationManager instance
        """
        from crewai_integration import CrewAIIntegrationManager
        return CrewAIIntegrationManager(api_base, api_key, project_id)
    
    def sync_subproblem_status_to_CrewAI(self, integration_manager, sub_problem_id: str, 
                                           new_status: str, solution_content: Optional[str] = None) -> bool:
        """
        Sync a specific sub-problem status to CrewAI
        
        Args:
            integration_manager: CrewAIIntegrationManager instance
            sub_problem_id: ID of the sub-problem to sync
            new_status: New status to set
            solution_content: Optional solution content to include
            
        Returns:
            True if sync was successful
        """
        return integration_manager.update_subproblem_status(self, sub_problem_id, new_status, solution_content)
    
    def sync_solution_to_CrewAI_ticket(self, integration_manager, sub_problem_id: str, 
                                         solution: 'SolutionAttempt') -> bool:
        """
        Sync a solution to its corresponding CrewAI ticket
        
        Args:
            integration_manager: CrewAIIntegrationManager instance
            sub_problem_id: ID of the sub-problem
            solution: SolutionAttempt to sync
            
        Returns:
            True if sync was successful
        """
        return integration_manager.sync_solution_to_ticket(self, sub_problem_id, solution)
    
    def sync_critique_to_CrewAI_ticket(self, integration_manager, sub_problem_id: str, 
                                         critique: 'CritiqueReport') -> bool:
        """
        Sync a critique report to its corresponding CrewAI ticket
        
        Args:
            integration_manager: CrewAIIntegrationManager instance
            sub_problem_id: ID of the sub-problem
            critique: CritiqueReport to sync
            
        Returns:
            True if sync was successful
        """
        return integration_manager.sync_critique_to_ticket(self, sub_problem_id, critique)
    
    def sync_verification_to_CrewAI_ticket(self, integration_manager, sub_problem_id: str, 
                                             verification: 'VerificationReport') -> bool:
        """
        Sync a verification report to its corresponding CrewAI ticket
        
        Args:
            integration_manager: CrewAIIntegrationManager instance
            sub_problem_id: ID of the sub-problem
            verification: VerificationReport to sync
            
        Returns:
            True if sync was successful
        """
        return integration_manager.sync_verification_to_ticket(self, sub_problem_id, verification)

    def update_subproblem_status(self, sub_problem_id: str, new_status: str):
        """
        Updates the status of a specific sub-problem within the decomposition plan.
        """
        if self.decomposition_plan:
            for sp in self.decomposition_plan.sub_problems:
                if sp.id == sub_problem_id:
                    sp.status = new_status
                    break
