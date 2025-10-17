# OpenEvolve: The Sovereign-Grade Decomposition Workflow

## Table of Contents

1.  [Overview & Guiding Principles](#10-overview--guiding-principles)
    *   1.1 Mission: Solving Intractable Problems
    *   1.2 Core Philosophy: Sovereign-Grade Control & Self-Healing Automation
2.  [Core Architecture: Teams & Gauntlets](#20-core-architecture-teams--gauntlets)
    *   2.1 The Team Abstraction
    *   2.2 The Gauntlet Abstraction
3.  [The End-to-End Workflow: A Microscopic Breakdown](#30-the-end-to-end-workflow-a-microscopic-breakdown)
    *   3.1 Stage 0: Content Analysis
    *   3.2 Stage 1: AI-Assisted Decomposition
    *   3.3 Stage 2: Manual Review & Override (The 'Command' Step)
    *   3.4 Stage 3: Sub-Problem Solving Loop
    *   3.5 Stage 4: Configurable Reassembly
    *   3.6 Stage 5: Final Verification & Self-Healing Loop
4.  [UI/UX Configuration Concept](#40-uiux-configuration-concept)
    *   4.1 The Team Manager
    *   4.2 The Gauntlet Designer
    *   4.3 The Workflow Orchestrator
5.  [Data Object Schemas](#50-data-object-schemas)
    *   5.1 `Team`
    *   5.2 `GauntletDefinition`
    *   5.3 `DecompositionPlan`
    *   5.4 `SolutionAttempt`
    *   5.5 `CritiqueReport` & `VerificationReport`

---

## 1.0 Overview & Guiding Principles

### 1.1 Mission: Solving Intractable Problems

The Sovereign-Grade Decomposition Workflow (SGDW) is designed to tackle complex, seemingly intractable problems by treating them not as a single challenge, but as a system of interconnected, solvable components. By applying rigorous, multi-agent AI strategies at every step, the SGDW can navigate vast solution spaces and produce highly reliable and verified results.

### 1.2 Core Philosophy: Sovereign-Grade Control & Self-Healing Automation

The workflow is built on two key principles:

1.  **Sovereign-Grade Control**: The user (the "Sovereign") has ultimate, microscopic control over every agent, process, and decision. The system provides intelligent defaults and suggestions, but the user has the final say.
2.  **Self-Healing Automation**: When faced with failures, the system is designed to intelligently diagnose the root cause, and automatically trigger targeted, recursive correction loops until a satisfactory solution is achieved.

---

## 2.0 Core Architecture: Teams & Gauntlets

### 2.1 The Team Abstraction

A **Team** is a user-defined, named group of AI models assigned to a specific role. This is the fundamental unit of action in the workflow.

*   **Team Roles**:
    *   **Blue Teams**: Responsible for creation and synthesis. Sub-roles include:
        *   `Planners`: Generate decomposition strategies.
        *   `Solvers`: Generate solutions for sub-problems.
        *   `Patchers`: Fix solutions rejected by Red or Gold teams.
        *   `Assemblers`: Integrate verified solutions into a final product.
    *   **Red Teams (`Assailants`)**: Responsible for criticism and flaw detection. They test the robustness of solutions.
    *   **Gold Teams (`Judges`)**: Responsible for impartial evaluation and scoring against defined criteria.

*   **Team Composition**: A team is a collection of specific model configurations (e.g., `gpt-4-turbo` with temp 0.5, `claude-3-opus` with temp 0.6). This allows for the creation of diverse, specialist teams.

### 2.2 The Gauntlet Abstraction

A **Gauntlet** is a programmable, multi-round process that a piece of content (e.g., a solution candidate) must pass. Each Gauntlet is run by a specific **Team**. The rules for a Gauntlet are fully configurable.

*   **Programmable Rules**:
    *   **Flexible Quorums**: Success can be defined as `M out of N` agents agreeing (e.g., 2 of 3 judges approve).
    *   **Per-Agent Requirements**: Different models within a team can have different rules (e.g., Judge A must give a score > 9.0, while Judge B only needs > 7.5).
    *   **Multi-Round Logic**: Each round in a gauntlet can have different rules (e.g., Round 1 requires a simple majority, but Round 2 requires unanimity).
    *   **Per-Agent Approval Counts**: Success can require a specific agent to approve a solution a certain number of times across all rounds.
    *   **Statistical Thresholds**: Success can depend on metrics like `score_variance` being below a certain threshold to ensure strong consensus.

---

## 3.0 The End-to-End Workflow: A Microscopic Breakdown

### 3.1 Stage 0: Content Analysis

*   **Input**: The user's raw, high-level problem description.
*   **Process**: A dedicated 'Content Analyzer' model performs a deep scan.
    1.  **Lexical & Syntactic Parsing**: Understands the structure of the request.
    2.  **Semantic Domain Identification**: Determines if the problem is related to code, law, science, etc.
    3.  **Metadata & Context Extraction**: Identifies key entities, constraints, and goals mentioned in the problem.
*   **Output**: An `AnalyzedContext` object containing structured information that will be used to generate more effective prompts in all subsequent stages.

### 3.2 Stage 1: AI-Assisted Decomposition

*   **Input**: `AnalyzedContext` object.
*   **Process**: A **Blue Team** (role: `Planner`) is invoked. Its prompt instructs it to generate a detailed, structured (JSON) decomposition plan.
*   **Output**: A `DecompositionPlan` object. This object contains a list of `SubProblem`s, each with:
    *   `id`: A unique identifier (e.g., "sub_1.1").
    *   `description`: The detailed sub-problem statement.
    *   `dependencies`: A list of other `id`s it depends on.
    *   `ai_suggestions`: An object containing AI-generated hints for the user, such as a suggested `evolution_mode`, a `complexity_score`, and a draft `evaluation_prompt`.

### 3.3 Stage 2: Manual Review & Override (The 'Command' Step)

*   **Input**: `DecompositionPlan` object.
*   **Process**: The plan is rendered in an interactive UI. The user can click into any sub-problem and edit every field. They can change dependencies, rewrite descriptions, and, most importantly, assign the specific **Gauntlets** to be used for solving and verifying that sub-problem.
*   **Output**: An `ApprovedPlan` object, which is structurally identical to the `DecompositionPlan` but contains the user's final configurations.

### 3.4 Stage 3: Sub-Problem Solving Loop

*   **Input**: The `ApprovedPlan`. The system uses a topological sort to process the sub-problems in an order that respects dependencies.
*   **Loop for each sub-problem**:
    1.  **Generation (Blue Team)**: The assigned 'Solvers' **Blue Team** runs its configured Gauntlet to produce a `SolutionAttempt`. This might involve an internal peer-review process as defined in the Gauntlet.
    2.  **Critique (Red Team Gauntlet)**: The `SolutionAttempt` is passed to the assigned 'Assailants' **Red Team**. The Red Team Gauntlet runs, applying its configured `attack modes`.
        *   **If Flaw is Found**: The solution is rejected. A `CritiqueReport` is generated. The process loops back to Step 1, but this time a 'Patchers' **Blue Team** is invoked, and it receives both the original sub-problem and the `CritiqueReport` to inform its work.
    3.  **Verification (Gold Team Gauntlet)**: If the solution survives the Red Team, it is passed to the assigned 'Judges' **Gold Team**. The Gold Team Gauntlet runs its programmable evaluation.
        *   **If Rejected**: The solution is rejected. A `VerificationReport` is generated. The process loops back to Step 1, invoking the 'Patchers' team with the report.
    4.  **Success**: If the solution passes both the Red and Gold Gauntlets, it is marked as `Verified` and its `VerifiedSolution` object is stored.

### 3.5 Stage 4: Configurable Reassembly

*   **Input**: The collection of `VerifiedSolution` objects for all sub-problems.
*   **Process**: The user-designated 'Assemblers' **Blue Team** is invoked. Its prompt instructs it to integrate all the verified components into a single, coherent final product, respecting the original problem statement and dependencies.
*   **Output**: A `FinalSolutionCandidate` object.

### 3.6 Stage 5: Final Verification & Self-Healing Loop

*   **Input**: `FinalSolutionCandidate`.
*   **Process**: The final candidate must pass its own two final, user-configured Gauntlets.
    1.  **Final Red Team Gauntlet**: Checks for integration errors, inconsistencies, or new vulnerabilities introduced during assembly.
    2.  **Final Gold Team Gauntlet**: Performs a holistic evaluation of the final product against the original high-level problem.
*   **Self-Healing Logic**:
    *   **If Failure**: The final Gauntlet is configured to require the Gold Team to provide **targeted feedback**, identifying which part of the solution (and therefore which original sub-problem) is the likely cause of the failure.
    *   The system parses this feedback, flags the corresponding sub-problem(s) as "requiring rework", and sends them back to **Stage 3** for a new solving loop.
    *   Once the reworked sub-problems are re-verified, the workflow proceeds to Stage 4 for reassembly, and the new final candidate is submitted to Stage 5 again.
    *   This continues until the final product is approved or a `max_refinement_loops` limit is hit, at which point the user is alerted for manual intervention.
*   **Output**: The final, `VerifiedFinalSolution`.

---

## 4.0 UI/UX Configuration Concept

### 4.1 The Team Manager

A dedicated screen where the user can visually create and manage their AI teams.
*   Create a new team, give it a name (e.g., "My Expert Code Judges"), and assign it a role (Blue, Red, or Gold).
*   Add models to the team from a list of available APIs, setting model-specific parameters (temp, top-p, etc.) for each.

### 4.2 The Gauntlet Designer

A UI for creating `GauntletDefinition` objects.
*   **Simple Mode**: Basic sliders and dropdowns for common cases (e.g., "Unanimous approval, 3 rounds").
*   **Advanced Mode**: A more complex interface (or raw JSON editor) for defining per-round and per-agent rules, quorums, and statistical thresholds.

### 4.3 The Workflow Orchestrator

The main screen for running the SGDW.
*   A dropdown to select the 'Sovereign-Grade Decomposition Workflow'.
*   Fields to input the initial problem.
*   Dropdowns to select the pre-configured **Teams** and **Gauntlets** for each step of the process (e.g., `Planner Team:`, `Solver Team:`, `Sub-Problem Red Gauntlet:`, `Final Judge Gauntlet:`, etc.).
*   An interactive 'Manual Review' panel that appears after Stage 1.
*   A real-time monitoring view that shows the progress of the entire workflow, including which sub-problem is being worked on and its status in the gauntlets.

---

## 5.0 Data Object Schemas

*   **5.1 `Team`**: `{ "name": "string", "role": "Blue|Red|Gold", "members": [ { "model_id": "string", "params": {...} } ] }`
*   **5.2 `GauntletDefinition`**: A list of `Round` objects, where each `Round` defines its `success_condition` (quorum, etc.) and `per_judge_requirements`.
*   **5.3 `DecompositionPlan`**: A list of `SubProblem` objects, each containing `id`, `description`, `dependencies`, and `gauntlet_assignments`.
*   **5.4 `SolutionAttempt`**: `{ "sub_problem_id": "string", "content": "string", "history": [...] }`
*   **5.5 `CritiqueReport` & `VerificationReport`**: `{ "solution_attempt_id": "string", "is_approved": "bool", "reports_by_judge": [ { "model_id": "string", "score": "float", "variance": "float", "justification": "string", "targeted_feedback": "string" } ] }`
