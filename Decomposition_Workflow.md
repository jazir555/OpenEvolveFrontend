Of course. Here is the comprehensive, combined version of the design document, integrating the structural improvements from the second document with the detailed content of the first.

***

# OpenEvolve: The Sovereign-Grade Decomposition Workflow - Full Design Document

## Table of Contents

1.  [Overview & Guiding Principles](#10-overview--guiding-principles)
    *   1.1 Mission: Solving Intractable Problems
    *   1.2 Core Philosophy: Sovereign-Grade Control & Self-Healing Automation
2.  [Core Architecture: Teams & Gauntlets](#20-core-architecture-teams--gauntlets)
    *   2.1 The Team Abstraction
        *   2.1.1 Team Roles (Blue, Red, Gold)
        *   2.1.2 Team Composition
    *   2.2 The Gauntlet Abstraction
        *   2.2.1 Programmable Rules
3.  [The End-to-End Workflow: A Microscopic Breakdown](#30-the-end-to-end-workflow-a-microscopic-breakdown)
    *   3.1 Stage 0: Content Analysis
    *   3.2 Stage 1: AI-Assisted Decomposition
    *   3.3 Stage 2: Manual Review & Override (The 'Command' Step)
    *   3.4 Stage 3: Sub-Problem Solving Loop
        *   3.4.1 Step A: Solution Generation (Blue Team)
        *   3.4.2 Step B: Critique (Red Team Gauntlet)
        *   3.4.3 Step C: Verification (Gold Team Gauntlet)
    *   3.5 Stage 4: Configurable Reassembly
    *   3.6 Stage 5: Final Verification & Self-Healing Loop
4.  [UI/UX Configuration Concept](#40-uiux-configuration-concept)
    *   4.1 The Team Manager
    *   4.2 The Gauntlet Designer
    *   4.3 The Workflow Orchestrator
    *   4.4 The Manual Review Panel
    *   4.5 The Real-time Monitoring View
5.  [Data Object Schemas (Detailed)](#50-data-object-schemas-detailed)
    *   5.1 `ModelConfig`
    *   5.2 `Team`
    *   5.3 `GauntletRoundRule`
    *   5.4 `GauntletDefinition`
    *   5.5 `SubProblem`
    *   5.6 `DecompositionPlan`
    *   5.7 `SolutionAttempt`
    *   5.8 `CritiqueReport`
    *   5.9 `VerificationReport`
    *   5.10 `WorkflowState`
6.  [Implementation Status & Remaining Tasks](#60-implementation-status--remaining-tasks)
    *   6.1 Completed Tasks (Phase 1, Phase 2, Phase 3)
    *   6.2 Remaining Tasks (Phase 4)

---

## 1.0 Overview & Guiding Principles

### 1.1 Mission: Solving Intractable Problems

The Sovereign-Grade Decomposition Workflow (SGDW) is designed to tackle complex, seemingly intractable problems by treating them not as a single challenge, but as a system of interconnected, solvable components. By applying rigorous, multi-agent AI strategies at every step, the SGDW can navigate vast solution spaces and produce highly reliable and verified results.

### 1.2 Core Philosophy: Sovereign-Grade Control & Self-Healing Automation

The workflow is built on two key principles:

1.  **Sovereign-Grade Control**: The user (the "Sovereign") has ultimate, microscopic control over every agent, process, and decision. The system provides intelligent defaults and suggestions, but the user has the final say. This includes defining AI teams, customizing evaluation criteria, and overriding AI-generated plans.
2.  **Self-Healing Automation**: When faced with failures, the system is designed to intelligently diagnose the root cause, and automatically trigger targeted, recursive correction loops until a satisfactory solution is achieved. This minimizes manual intervention while maximizing reliability.

---

## 2.0 Core Architecture: Teams & Gauntlets

### 2.1 The Team Abstraction

A **Team** is a user-defined, named group of AI models assigned to a specific role. This is the fundamental unit of action in the workflow.

#### 2.1.1 Team Roles (Blue, Red, Gold)

*   **Blue Teams**: Responsible for creation and synthesis. Their primary function is to generate, refine, and assemble content. Sub-roles include:
    *   `Planners`: Generate initial decomposition strategies and sub-problem definitions.
    *   `Solvers`: Generate initial solutions for individual sub-problems.
    *   `Patchers`: Analyze critique/verification reports and modify existing solutions to address identified flaws.
    *   `Assemblers`: Integrate verified solutions into a final, coherent product.
*   **Red Teams (`Assailants`)**: Responsible for criticism and flaw detection. They act as adversarial agents, actively seeking vulnerabilities, inconsistencies, and weaknesses in generated content.
*   **Gold Teams (`Judges`)**: Responsible for impartial evaluation and scoring against defined criteria. They verify the correctness, quality, and adherence to requirements of solutions.

#### 2.1.2 Team Composition

A team is a collection of specific `ModelConfig` objects. Each `ModelConfig` specifies an AI model (e.g., `gpt-4-turbo`, `claude-3-opus`), its API key, base URL, and generation parameters (temperature, top-p, max_tokens, etc.). This allows for the creation of diverse, specialist teams where each member can be fine-tuned for its specific task. Teams are created and managed via the 'Team Manager' UI (see section 4.1) and are defined by the `Team` data object (see section 5.2).

### 2.2 The Gauntlet Abstraction

A **Gauntlet** is a programmable, multi-round process that a piece of content (e.g., a solution candidate, a critique) must pass. Each Gauntlet is run by a specific **Team** (Blue, Red, or Gold). The rules for a Gauntlet are fully configurable, providing microscopic control over the evaluation process. Gauntlets are created via the 'Gauntlet Designer' UI (see section 4.2) and are defined by the `GauntletDefinition` data object (see section 5.4).

#### 2.2.1 Programmable Rules

*   **Flexible Quorums**: Define success for a round as `M out of N` agents agreeing (e.g., 2 of 3 judges approve). This moves beyond simple unanimity.
*   **Per-Agent Requirements**: Different models within a team can have different minimum score thresholds or other criteria for success in a given round.
*   **Multi-Round Logic**: Each round in a gauntlet can have distinct rules. For example, Round 1 might require a simple majority, while Round 2 demands unanimity.
*   **Per-Agent Approval Counts**: Success can require a specific agent to achieve a certain number of successful evaluations across all rounds of the gauntlet.
*   **Statistical Thresholds**: Gauntlets can incorporate statistical measures like `score_variance` to ensure strong consensus among judges, failing a solution if the variance is too high, even if average scores are good.
*   **Collaboration Modes**: Judges in later rounds can optionally be configured to see feedback from previous rounds or from other judges to facilitate consensus or challenge.

---

## 3.0 The End-to-End Workflow: A Microscopic Breakdown

The workflow proceeds through the following stages, with detailed inputs, processes, and outputs:

### 3.1 Stage 0: Content Analysis

*   **Purpose**: To thoroughly understand the user's initial problem statement and extract all relevant context before decomposition begins. This foundational step ensures that subsequent AI actions are well-informed and targeted.
*   **Input**: The user's raw, high-level problem description (string).
*   **Process**: A dedicated **Blue Team** (role: `Content Analyzer`) is invoked.
    1.  **Prompt Generation**: A specialized prompt is constructed, instructing the AI to act as a highly skilled content analyzer. This prompt is dynamically generated, incorporating best practices for LLM analysis.
    2.  **LLM Invocation**: The Content Analyzer team's models process the problem statement. The team can consist of multiple models for diverse perspectives, with their outputs potentially aggregated or cross-referenced.
    3.  **Structured Output**: The AI is instructed to provide its analysis in a structured JSON format. This ensures machine-readability and consistency. Key extracted fields include:
        *   `domain`: (e.g., "Software Development", "Physics", "Legal") - Categorizes the problem for specialized processing.
        *   `keywords`: List of important terms - Used for contextual prompting and search.
        *   `estimated_complexity`: (1-10) - An AI-generated initial assessment of the problem's difficulty.
        *   `potential_challenges`: List of anticipated difficulties - Helps in proactive planning.
        *   `required_expertise`: List of expertise areas needed - Guides selection of specialist AI models.
        *   `summary`: A brief, concise summary of the problem.
*   **Output**: An `AnalyzedContext` object (dictionary) containing structured information that will be used to generate more effective prompts in all subsequent stages.
*   **Configurability**: The Content Analyzer Team is user-selectable.

### 3.2 Stage 1: AI-Assisted Decomposition

*   **Purpose**: To break down the complex problem into a manageable set of sub-problems, complete with AI-suggested strategies for solving and evaluating each. This stage transforms an intractable problem into a structured plan.
*   **Input**: `AnalyzedContext` object.
*   **Process**: A **Blue Team** (role: `Planner`) is invoked.
    1.  **Prompt Generation**: A specialized prompt is constructed, instructing the AI to act as an expert problem decomposer, leveraging the `AnalyzedContext`.
    2.  **LLM Invocation**: The Planner team's models generate a detailed decomposition plan. Multiple models can contribute, with their outputs potentially synthesized for a more robust plan.
    3.  **Structured Output**: The AI is instructed to provide its output as a JSON array of `SubProblem` objects. Each `SubProblem` includes:
        *   `id`: A unique identifier (e.g., "sub_1.1", "sub_2.3") - Essential for tracking and dependency management.
        *   `description`: A clear, concise statement of the sub-problem to be solved.
        *   `dependencies`: A list of other `id`s of sub-problems this one depends on - Crucial for execution order.
        *   `ai_suggested_evolution_mode`: Suggested evolution mode (e.g., "standard", "adversarial", "quality_diversity") - AI's recommendation for the best approach to solve this specific sub-problem.
        *   `ai_suggested_complexity_score`: An integer from 1 to 10 - AI's estimate of the sub-problem's difficulty.
        *   `ai_suggested_evaluation_prompt`: A specific, tailored prompt for a Gold Team to evaluate this sub-problem's solution - Ensures relevant and precise verification.
*   **Output**: A `DecompositionPlan` object.
*   **Configurability**: The Planner Team is user-selectable.

### 3.3 Stage 2: Manual Review & Override (The 'Command' Step)

*   **Purpose**: To provide the user (the Sovereign) with microscopic control over the AI-generated decomposition plan, allowing for expert human intervention, refinement, and strategic decision-making before execution. This is the critical human-in-the-loop stage.
*   **Input**: `DecompositionPlan` object.
*   **Process**:
    1.  **UI Rendering**: The `DecompositionPlan` is rendered in an interactive Streamlit UI panel (`render_manual_review_panel`). This panel presents the plan in an easily digestible and editable format.
    2.  **User Interaction**: The user can meticulously review and modify every aspect of the plan:
        *   **Sub-Problem Details**: Edit `description`, `dependencies`, `ai_suggested_evolution_mode`, `ai_suggested_complexity_score`, and `ai_suggested_evaluation_prompt` for any sub-problem.
        *   **Team & Gauntlet Assignment**: **Crucially**, the user can override AI suggestions and assign specific **Gauntlets** (Red and Gold) and **Blue Teams** (Solvers, Patchers) to each individual sub-problem. This allows for highly specialized teams and rigorous verification tailored to each component.
        *   **Specific Evolution Parameters**: Provide a JSON object for `evolution_params` to fine-tune the underlying evolution process for a particular sub-problem.
        *   **Approval/Rejection**: The user explicitly approves or rejects the entire plan.
    3.  **State Management**: The workflow pauses, awaiting user input. Streamlit's session state manages this interactive pause, ensuring continuity.
*   **Output**: An `ApprovedPlan` object, which is structurally identical to the `DecompositionPlan` but contains the user's final, approved configurations for each sub-problem. If rejected, the workflow terminates or prompts for re-initiation.
*   **Configurability**: User-driven, providing the highest level of control.

### 3.4 Stage 3: Sub-Problem Solving Loop

*   **Purpose**: To iteratively generate, critique, and verify solutions for each sub-problem according to the `ApprovedPlan`, respecting dependencies and applying self-healing mechanisms.
*   **Input**: The `ApprovedPlan` object.
*   **Process**:
    1.  **Dependency Management**: The system processes sub-problems in an order that respects their dependencies. A topological sort ensures that a sub-problem is only attempted after all its prerequisites are met.
    2.  **Iterative Loop for each sub-problem (until Verified or max retries reached)**:
        *   **a. Solution Generation (Blue Team)**:
            *   The assigned 'Solvers' **Blue Team** is invoked.
            *   It runs its configured Gauntlet (e.g., `single_candidate` for direct generation, or `multi_candidate_peer_review` for internal refinement) to produce a `SolutionAttempt` for the current sub-problem.
        *   **b. Critique (Red Team Gauntlet)**:
            *   The `SolutionAttempt` is passed to the assigned 'Assailants' **Red Team**.
            *   The Red Team Gauntlet runs, applying its configured `attack modes` (e.g., 'Security Scan', 'Edge Case Exploration', 'Assumption Challenge').
            *   **If Flaw is Found (Gauntlet Fails)**: The solution is rejected. A `CritiqueReport` is generated, detailing the flaws. The process loops back to Step 3.4.1. This time, the assigned 'Patchers' **Blue Team** is invoked, receiving the original sub-problem and the `CritiqueReport` to inform its work, aiming to fix the identified issues.
        *   **c. Verification (Gold Team Gauntlet)**:
            *   If the solution survives the Red Team, it is passed to the assigned 'Judges' **Gold Team**.
            *   The Gold Team Gauntlet runs its programmable evaluation, using the `evaluation_prompt` specific to this sub-problem. The Gauntlet's rules (quorum, per-agent, multi-round, statistical thresholds like `score_variance`) determine if the solution is verified.
            *   **If Rejected (Gauntlet Fails)**: The solution is rejected. A `VerificationReport` is generated. The process loops back to Step 3.4.1, invoking the 'Patchers' team with the report.
        *   **Success**: If the solution passes both the Red and Gold Gauntlets, it is marked as `Verified` and its `VerifiedSolution` object is stored in `workflow_state.sub_problem_solutions`.
*   **Output**: A collection of `VerifiedSolution` objects for all sub-problems.
*   **Configurability**: User-selectable Solver Team, Patcher Team, Sub-Problem Red Gauntlet, and Sub-Problem Gold Gauntlet for each sub-problem.

### 3.5 Stage 4: Configurable Reassembly

*   **Purpose**: To integrate all individually verified sub-problem solutions into a single, coherent final product. This stage focuses on the synthesis of components.
*   **Input**: The collection of `VerifiedSolution` objects for all sub-problems.
*   **Process**: The user-designated 'Assemblers' **Blue Team** is invoked. Its prompt instructs it to integrate all the verified components into a single, coherent final product, respecting the original problem statement and dependencies. The Assembler Team can also run its own Gauntlet for internal quality checks before outputting the final candidate.
*   **Output**: A `FinalSolutionCandidate` object.
*   **Configurability**: User-selectable Assembler Team.

### 3.6 Stage 5: Final Verification & Self-Healing Loop

*   **Purpose**: To rigorously verify the final assembled solution and, if necessary, trigger targeted self-correction until the solution meets all criteria, ensuring the integrity of the entire solution.
*   **Input**: `FinalSolutionCandidate`.
*   **Process**: The final candidate must pass its own two final, user-configured Gauntlets.
    1.  **Final Red Team Gauntlet**: The assembled solution is subjected to a final adversarial attack by the designated Red Team. This checks for integration errors, inconsistencies, or new vulnerabilities that may have arisen from the assembly process.
    2.  **Final Gold Team Gauntlet**: The solution is then holistically evaluated by the designated Gold Team against the original high-level problem statement and overall quality criteria. This Gauntlet uses its own programmable rules for quorum, rounds, and statistical thresholds.
*   **Self-Healing Logic**:
    *   **If Failure**: The final Gauntlet is configured to require the Gold Team to provide **targeted feedback**, identifying which part of the solution (and therefore which original sub-problem) is the likely cause of the failure.
    *   The system parses this feedback, flags the corresponding sub-problem(s) as "requiring rework", and sends them back to **Stage 3** for a new solving loop.
    *   Once the reworked sub-problems are re-verified, the workflow proceeds to Stage 4 for reassembly, and the new `FinalSolutionCandidate` is submitted to Stage 5 again.
    *   This continues until the final product is approved or a `max_refinement_loops` limit is hit, at which point the user is alerted for manual intervention.
*   **Output**: The final, `VerifiedFinalSolution` object.
*   **Configurability**: User-selectable Final Red Team Gauntlet, Final Gold Team Gauntlet, and `max_refinement_loops`.

---

## 4.0 UI/UX Configuration Concept

This section outlines the user interface components that will enable the Sovereign to configure, monitor, and interact with the workflow.

### 4.1 The Team Manager

*   **Location**: Accessible via the "Configuration" tab in the main Orchestrator UI.
*   **Functionality**: Allows users to visually create and manage their AI teams.
    *   **Team Creation Form**: Provides input fields for `Team Name`, `Team Role` (Blue, Red, or Gold), and a `Description` for the team's purpose.
    *   **Model Configuration**: Features dynamic forms to add multiple `ModelConfig` entries to a team, specifying `model_id`, `api_key`, `api_base`, and generation parameters like `temperature`, `top-p`, `max_tokens`, `frequency_penalty`, `presence_penalty`, and `seed`.
    *   **Team List**: Displays all created teams in an organized manner, with options to expand each team entry to view its members, edit its configuration, or delete the team.

### 4.2 The Gauntlet Designer

*   **Location**: Accessible via the "Configuration" tab in the main Orchestrator UI.
*   **Functionality**: Provides a structured interface for creating and managing `GauntletDefinition` objects, which define the programmable evaluation processes.
    *   **Gauntlet Creation Form**: Input fields for `Gauntlet Name`, `Description`, and a crucial dropdown to select the `Team` that will run this specific gauntlet.
    *   **Round Configuration**: Features dynamic forms to add and configure multiple `GauntletRoundRule` definitions. For each round, users can specify:
        *   `Round Number`.
        *   `Quorum: Required Approvals` (e.g., 2) and `Quorum: From Panel Size` (e.g., 3) to define the success threshold.
        *   `Minimum Overall Confidence` (a slider from 0.0-1.0) for the average score across all judges in that round.
        *   `Maximum Score Variance` (optional, numeric input) to ensure consensus among judges.
        *   `Per-Judge Requirements`: An advanced JSON text area where users can specify `min_score` or `required_successful_rounds` for individual models within the panel, offering microscopic control.
        *   `Collaboration Mode` (dropdown: "independent" or "share_previous_feedback") to control information flow between judges.
    *   **Team-Specific Settings**: Additional fields appear based on the selected team's role:
        *   For Red Team Gauntlets: `Red Team Attack Modes` (comma-separated input for specific adversarial techniques).
        *   For Blue Team Gauntlets: `Blue Team Generation Mode` (dropdown: "single_candidate" or "multi_candidate_peer_review").
    *   **Gauntlet List**: Displays all created gauntlets, with options to expand to view their detailed round rules, edit their configurations, or delete them.

### 4.3 The Workflow Orchestrator

*   **Location**: The main "Create Workflow" tab.
*   **Functionality**: The central control panel for initiating and configuring the SGDW, allowing the user to assemble a complete problem-solving pipeline.
    *   **Workflow Type Selection**: A prominent dropdown including "👑 Sovereign-Grade Decomposition" as a selectable option.
    *   **Problem Input**: A large text area for the initial problem statement that the user wants to solve.
    *   **Team/Gauntlet Selection**: For the "Sovereign-Grade Decomposition" workflow, a series of dropdowns will dynamically appear, allowing the user to select pre-configured **Teams** and **Gauntlets** for each critical step of the process:
        *   `Content Analyzer Team` (Blue)
        *   `Planner Team` (Blue)
        *   `Solver Team` (Blue)
        *   `Patcher Team` (Blue)
        *   `Sub-Problem Red Team Gauntlet`
        *   `Sub-Problem Gold Team Gauntlet`
        *   `Assembler Team` (Blue)
        *   `Final Red Team Gauntlet`
        *   `Final Gold Team Gauntlet`
        *   `Max Refinement Loops` (numeric input) - Configures the self-healing mechanism.
    *   **Start Workflow Button**: Initiates the workflow, storing the complete configuration in `st.session_state.active_sovereign_workflow` and triggering the execution process.

### 4.4 The Manual Review Panel

*   **Location**: Appears dynamically in the main content area when `workflow_state.current_stage` is "Manual Review & Override".
*   **Functionality**: Presents the AI-generated `DecompositionPlan` for user inspection and modification, serving as the critical human-in-the-loop control point.
    *   Displays the overall problem statement and a summary of the analyzed context.
    *   Lists each sub-problem in an expandable section.
    *   **Editable Fields**: For each sub-problem, the user can directly edit:
        *   `Description` of the sub-problem.
        *   `Dependencies` (comma-separated IDs).
        *   `AI Suggested Evolution Mode`, `Complexity Score`, `Evaluation Prompt` (these are AI suggestions but are fully editable).
        *   **User Overrides**: Dropdowns to select specific `Solver Team`, `Red Team Gauntlet`, and `Gold Team Gauntlet` for that particular sub-problem, overriding any AI suggestions.
        *   A JSON text area for `Specific Evolution Parameters` to fine-tune the underlying evolution process for that sub-problem.
    *   **Action Buttons**: "✅ Approve Plan" (proceeds to Stage 3 with the modified plan) and "❌ Reject Plan" (terminates the workflow or prompts for re-initiation).

### 4.5 The Real-time Monitoring View

*   **Location**: Appears dynamically in the "Monitoring Panel" tab when a Sovereign-Grade workflow is active.
*   **Functionality**: Provides live, granular updates on the workflow's progress, allowing the user to track the entire self-healing process.
    *   Displays `Workflow ID`, `Current Stage`, `Current Sub-Problem ID` (if applicable), `Current Gauntlet Name` (if applicable).
    *   A visual progress bar for the overall workflow.
    *   Status messages (e.g., "Analyzing problem statement...", "Running Red Team Gauntlet for sub_1.2...", "Awaiting user approval...").
    *   Automatically triggers `st.rerun()` to continue workflow execution and update the display, providing a seamless, interactive experience.
    *   Displays final success/failure messages, along with links to detailed reports.

---

## 5.0 Data Object Schemas (Detailed)

These are the Python `dataclasses` defined in `workflow_structures.py`, serving as the backbone for data management. They ensure type safety, clarity, and ease of serialization/deserialization.

### 5.1 `ModelConfig`

```python
@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single AI model within a team."""
    model_id: str  # Unique identifier for the AI model (e.g., "gpt-4o", "claude-3-opus")
    api_key: str  # API key for authentication with the model provider
    api_base: str = "https://api.openai.com/v1"  # Base URL for the API endpoint
    temperature: float = 0.7  # Controls randomness in model outputs (0.0-2.0)
    top_p: float = 1.0  # Nucleus sampling parameter (0.0-1.0)
    max_tokens: int = 4096  # Maximum number of tokens to generate
    frequency_penalty: float = 0.0  # Penalizes new tokens based on their existing frequency in the text so far
    presence_penalty: float = 0.0  # Penalizes new tokens based on whether they appear in the text so far
    seed: Optional[int] = None  # Seed for reproducible sampling
    # Additional model-specific parameters can be added here.
```

### 5.2 `Team`

```python
@dataclasses.dataclass
class Team:
    """A user-defined group of AI models assigned to a specific role."""
    name: str  # Unique name for the team
    role: Literal["Blue", "Red", "Gold"]  # Specifies the team's primary function
    members: List[ModelConfig]  # List of AI models comprising the team
    description: Optional[str] = None  # Human-readable description of the team's purpose
```

### 5.3 `GauntletRoundRule`

```python
@dataclasses.dataclass
class GauntletRoundRule:
    """Defines the specific rules and criteria for a single round within a Gauntlet."""
    round_number: int  # The sequential number of this round within the gauntlet
    # Quorum for the round: M out of N judges must approve
    quorum_required_approvals: int  # Minimum number of team members that must approve for this round to pass
    quorum_from_panel_size: int  # Total number of team members participating in this round (typically len(Team.members))
    
    # Overall confidence threshold for the round
    min_overall_confidence: float = 0.0  # e.g., 0.75 for 75% average score across all judges in this round
    
    # Optional: Statistical thresholds for consensus among judges
    max_score_variance: Optional[float] = None  # e.g., 0.1 to ensure judges' scores are tightly clustered; if variance exceeds this, the round fails
    
    # Per-judge requirements for this round (overrides global round rules for specific models)
    # Example: {"gemini-pro": {"min_score": 0.9, "required_successful_rounds": 3}}
    # The key is the model_id, value is a dict of specific requirements for that model in this round.
    per_judge_requirements: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    
    # Optional: Collaboration mode for judges in this round
    collaboration_mode: Literal["independent", "share_previous_feedback"] = "independent"
    # "independent": Judges evaluate without seeing others' feedback.
    # "share_previous_feedback": Judges in later rounds can see feedback from earlier rounds/judges to facilitate consensus or challenge.
```

### 5.4 `GauntletDefinition`

```python
@dataclasses.dataclass
class GauntletDefinition:
    """A programmable, multi-round process that a piece of content must pass to be approved."""
    name: str  # Unique name for the gauntlet
    team_name: str  # Name of the Team that runs this Gauntlet
    rounds: List[GauntletRoundRule]  # Ordered list of rules for each round of the gauntlet
    description: Optional[str] = None  # Human-readable description of the gauntlet's purpose
    
    # For Red Team Gauntlets: specific attack modes to guide the AI's critique
    attack_modes: List[str] = dataclasses.field(default_factory=list)  # e.g., ["Security Scan", "Edge Case Analysis", "Assumption Challenge"]
    
    # For Blue Team Gauntlets: defines how solutions are generated/reviewed internally
    generation_mode: Literal["single_candidate", "multi_candidate_peer_review"] = "single_candidate"
    # "single_candidate": One model generates one solution attempt.
    # "multi_candidate_peer_review": Multiple models generate candidates, then another model synthesizes/reviews them into a single, best candidate.
```

### 5.5 `SubProblem`

```python
@dataclasses.dataclass
class SubProblem:
    """Represents a single sub-problem in the decomposition plan, with its own configurations."""
    id: str  # Unique identifier (e.g., "sub_1.1", "sub_2.3")
    description: str  # Detailed statement of the sub-problem to be solved
    dependencies: List[str] = dataclasses.field(default_factory=list)  # IDs of other sub-problems it depends on
    
    # AI suggestions (generated in Stage 1, can be overridden in Stage 2)
    ai_suggested_evolution_mode: str = "standard"  # e.g., "standard", "adversarial"
    ai_suggested_complexity_score: int = 5  # AI's estimate of complexity (1-10)
    ai_suggested_evaluation_prompt: str = ""  # AI's suggested prompt for Gold Team evaluation of this sub-problem
    
    # User-approved configurations (from Stage 2)
    solver_team_name: str = ""  # Name of the Blue Team assigned to solve this sub-problem
    red_team_gauntlet_name: Optional[str] = None  # Name of the Red Team Gauntlet to critique this sub-problem's solution
    gold_team_gauntlet_name: str = ""  # Name of the Gold Team Gauntlet to verify this sub-problem's solution
    
    # Specific evolution parameters for this sub-problem (can override global settings for the solver)
    evolution_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
```

### 5.6 `DecompositionPlan`

```python
@dataclasses.dataclass
class DecompositionPlan:
    """The overall plan for decomposing and solving a complex problem, including global configurations."""
    problem_statement: str  # The original problem provided by the user
    analyzed_context: Dict[str, Any]  # Output from Stage 0 (Content Analysis)
    sub_problems: List[SubProblem]  # List of all sub-problems with their configurations
    
    # Global workflow configurations (can be set in UI)
    max_refinement_loops: int = 3  # Max iterations for the self-healing loop in Stage 5
    
    # Teams and Gauntlets for final stages (user-selected in UI)
    assembler_team_name: str = ""  # Name of the Blue Team for reassembly of the final solution
    final_red_team_gauntlet_name: Optional[str] = None  # Name of the Red Team Gauntlet for the final product
    final_gold_team_gauntlet_name: str = ""  # Name of the Gold Team Gauntlet for the final product
```

### 5.7 `SolutionAttempt`

```python
@dataclasses.dataclass
class SolutionAttempt:
    """Represents a candidate solution for a sub-problem or the final solution at a given point in time."""
    sub_problem_id: str  # ID of the sub-problem this solution is for (or "final_solution" for the main product)
    content: str  # The actual generated solution (code, text, etc.)
    generated_by_model: str  # Which specific model generated this attempt
    timestamp: float  # Unix timestamp when this attempt was generated
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)  # To track changes/iterations if applicable
```

### 5.8 `CritiqueReport`

```python
@dataclasses.dataclass
class CritiqueReport:
    """Report generated by a Red Team Gauntlet, detailing identified flaws and overall robustness."""
    solution_attempt_id: str  # ID of the solution attempt being critiqued
    gauntlet_name: str  # Name of the Red Team Gauntlet that ran
    is_approved: bool  # True if it passed the Red Team (i.e., NO critical flaws found, solution is robust)
    reports_by_judge: List[Dict[str, Any]]  # Detailed reports from each Red Team member, including score, justification, and targeted feedback
    summary: str = ""  # Overall summary of the critique process
```

### 5.9 `VerificationReport`

```python
@dataclasses.dataclass
class VerificationReport:
    """Report generated by a Gold Team Gauntlet, detailing verification results and confidence."""
    solution_attempt_id: str  # ID of the solution attempt being verified
    gauntlet_name: str  # Name of the Gold Team Gauntlet that ran
    is_approved: bool  # True if it passed the Gold Team's verification criteria
    reports_by_judge: List[Dict[str, Any]]  # Detailed reports from each Gold Team member, including score, justification, and targeted feedback
    average_score: float = 0.0  # Average confidence score across all judges in the final round
    score_variance: float = 0.0  # Variance of scores, indicating consensus among judges
    summary: str = ""  # Overall summary of the verification process
```

### 5.10 `WorkflowState`

```python
@dataclasses.dataclass
class WorkflowState:
    """Manages the dynamic state of an active Sovereign-Grade Decomposition Workflow run."""
    workflow_id: str  # Unique ID for this workflow run
    problem_statement: str  # The initial problem statement for this workflow
    current_stage: str  # Current stage of the workflow (e.g., "Content Analysis", "Manual Review & Override", "Sub-Problem Solving Loop")
    current_sub_problem_id: Optional[str] = None  # ID of the sub-problem currently being processed
    current_gauntlet_name: Optional[str] = None  # Name of the gauntlet currently running
    status: str = "running"  # Overall status: "running", "paused", "completed", "failed", "awaiting_user_input"
    progress: float = 0.0  # 0.0 to 1.0, overall progress indicator for the workflow
    start_time: float = dataclasses.field(default_factory=time.time)  # Unix timestamp when workflow started
    end_time: Optional[float] = None  # Unix timestamp when workflow ended
    
    decomposition_plan: Optional[DecompositionPlan] = None  # The AI-generated/user-approved plan for this workflow
    sub_problem_solutions: Dict[str, SolutionAttempt] = dataclasses.field(default_factory=dict)  # Stores verified solutions for each sub-problem
    final_solution: Optional[SolutionAttempt] = None  # The final assembled solution attempt
    
    refinement_loop_count: int = 0  # Counter for the self-healing loop in Stage 5
    
    # Store all critique and verification reports for auditing and debugging
    all_critique_reports: List[CritiqueReport] = dataclasses.field(default_factory=list)
    all_verification_reports: List[VerificationReport] = dataclasses.field(default_factory=list)

    # Store the specific teams and gauntlets used for THIS workflow run.
    # This ensures consistency even if global definitions in TeamManager/GauntletManager change later.
    content_analyzer_team: Optional[Team] = None
    planner_team: Optional[Team] = None
    solver_team: Optional[Team] = None
    patcher_team: Optional[Team] = None
    assembler_team: Optional[Team] = None
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None
    final_red_gauntlet: Optional[GauntletDefinition] = None
    final_gold_gauntlet: Optional[GauntletDefinition] = None
    max_refinement_loops: int = 3 # Max iterations for the self-healing loop
```

---

## 6.0 Implementation Status & Remaining Tasks

This section details the current progress of the implementation based on the `TODO.md` file, providing a granular overview of completed and pending tasks.

### 6.1 Completed Tasks (Phase 1, Phase 2, Phase 3)

The following components and functionalities have been successfully implemented:

*   **Phase 1: Core Structures & Configuration UI**
    *   **`workflow_structures.py`**: Created and refined. This file defines all the core data objects (`ModelConfig`, `Team`, `GauntletRoundRule`, `GauntletDefinition`, `SubProblem`, `DecompositionPlan`, `SolutionAttempt`, `CritiqueReport`, `VerificationReport`, `WorkflowState`) that underpin the entire workflow. Docstrings and comments have been added for clarity.
    *   **`team_manager.py`**: Created and refined. This module provides the logic for persistent storage (using JSON files) and management (CRUD operations) of `Team` objects. Docstrings and comments have been added.
    *   **`gauntlet_manager.py`**: Created and refined. Similar to `team_manager.py`, this module handles the persistent storage and management of `GauntletDefinition` objects. Docstrings and comments have been added.
    *   **`ui_components.py`**: Created. This file houses the Streamlit UI functions for:
        *   `render_team_manager()`: Allows users to create, view, edit, and delete `Team` configurations.
        *   `render_gauntlet_designer()`: Allows users to create, view, edit, and delete `GauntletDefinition` objects, including defining complex round rules.
        *   `render_manual_review_panel()`: Provides an interactive UI for users to review and override AI-generated `DecompositionPlan`s.
    *   **`openevolve_orchestrator.py` (UI Integration)**: Modified to:
        *   Integrate `render_team_manager()` and `render_gauntlet_designer()` under the "Configuration" tab.
        *   Add the "👑 Sovereign-Grade Decomposition" workflow type to the `EvolutionWorkflow` enum and its UI representation.
        *   Include the UI for configuring a new Sovereign-Grade workflow, featuring dropdowns to select pre-configured Teams and Gauntlets for each stage.
        *   Update the "Start Workflow" button logic to correctly initiate the Sovereign-Grade workflow, storing its `WorkflowState` in Streamlit's session.

*   **Phase 2: Workflow Engine Implementation**
    *   **`workflow_engine.py`**: Created. This file contains the core logic for executing the workflow.
    *   `_request_openai_compatible_chat()`: Implemented as a robust utility for LLM API calls.
    *   `_compose_messages()`: Helper for structuring LLM prompts.
    *   `run_content_analysis()`: Implemented (Stage 0). Uses a Blue Team to analyze the problem statement.
    *   `run_ai_decomposition()`: Implemented (Stage 1). Uses a Blue Team (Planners) to generate the initial `DecompositionPlan`.
    *   `run_gauntlet()`: Implemented. This critical function interprets a `GauntletDefinition` and executes it with a given `Team`, applying programmable rules for each round and generating detailed reports (`CritiqueReport` or `VerificationReport`).
    *   `run_sovereign_workflow()`: The main orchestrator function has been implemented. It manages state transitions, calls the stage-specific functions, and includes the foundational logic for the self-healing loop.

*   **Phase 3: UI Integration & Interactivity**
    *   **`openevolve_orchestrator.py` (Workflow UI)**:
        *   The "Sovereign-Grade Decomposition Workflow" has been added to the list of available workflow types.
        *   The UI for configuring a new workflow, including dropdowns to select pre-configured Teams and Gauntlets for each stage, has been created.
        *   The "Manual Review" panel (`render_manual_review_panel`) has been implemented in `ui_components.py` and is ready to be dynamically called by the orchestrator.
        *   The real-time monitoring view for the workflow's progress has been implemented in `openevolve_orchestrator.py`, dynamically displaying the `WorkflowState` and triggering `run_sovereign_workflow` for continuous execution.

### 6.2 Remaining Tasks (Phase 4)

The following tasks are crucial for completing the full implementation:

*   **Implement the "Manual Review" panel's dynamic invocation**: The `run_sovereign_workflow` currently simulates approval for Stage 2. The orchestrator needs to dynamically render `render_manual_review_panel` and pause execution until the user approves the plan. This requires careful Streamlit state management to handle the interactive pause.
*   **Refine `generate_solution_for_sub_problem`**: The current implementation is a placeholder. This needs to be replaced with actual logic for generating solutions, potentially integrating with existing OpenEvolve evolution loops or other generation mechanisms based on the `SubProblem`'s `ai_suggested_evolution_mode` and `evolution_params`.
*   **Refine `parse_targeted_feedback`**: The current implementation uses a simple regex. This needs to be enhanced to robustly parse structured feedback (e.g., JSON) from LLM reports to accurately identify problematic sub-problem IDs for the self-healing loop.
*   **Implement Blue Team Gauntlet for Generation/Peer Review**: The `run_gauntlet` function has a placeholder for Blue Team gauntlets. This needs to be fully implemented to support `single_candidate` and `multi_candidate_peer_review` generation modes.
*   **Full Review of Docstrings and Comments**: While initial docstrings and comments are present, a final pass is needed to ensure every function, class, and complex logic block is thoroughly documented.
*   **Comprehensive Integration Testing**: End-to-end testing of the entire workflow, including all gauntlets, self-healing loops, and UI interactions, is essential.
*   **Error Handling and Edge Cases**: Implement more robust error handling and consider edge cases (e.g., no teams/gauntlets defined, circular dependencies in sub-problems).
*   **Performance Optimization**: As a "Sovereign-Grade" system, performance will be critical. This includes optimizing LLM calls (parallelization, caching), Streamlit rendering, and data persistence.
*   **Remove Placeholders**: All `st.warning("Placeholder: ...")` and similar temporary code must be replaced with production-ready implementations.