import json
import os
from typing import Dict, List, Any
from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, SolutionAttempt, CritiqueReport, VerificationReport, ModelConfig, Team, GauntletDefinition, GauntletRoundRule
import dataclasses

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle dataclass objects.
    """
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

class WorkflowHistoryManager:
    """
    Manages the persistent storage and retrieval of WorkflowState objects.
    Stores workflow history in a JSON file.
    """
    def __init__(self, history_file: str = "workflow_history.json"):
        self.history_file = os.path.join(os.getcwd(), history_file)
        self._load_history()

    def _load_history(self) -> None:
        """
        Loads workflow history from the JSON file.
        """
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                try:
                    raw_history = json.load(f)
                    self.history: Dict[str, WorkflowState] = {}
                    for wf_id, wf_data in raw_history.items():
                        # Reconstruct WorkflowState and its nested dataclasses
                        # This requires careful handling of nested dataclasses
                        try:
                            # Reconstruct ModelConfig
                            if 'content_analyzer_team' in wf_data and wf_data['content_analyzer_team']:
                                wf_data['content_analyzer_team']['members'] = [ModelConfig(**m) for m in wf_data['content_analyzer_team']['members']]
                                wf_data['content_analyzer_team'] = Team(**wf_data['content_analyzer_team'])
                            if 'planner_team' in wf_data and wf_data['planner_team']:
                                wf_data['planner_team']['members'] = [ModelConfig(**m) for m in wf_data['planner_team']['members']]
                                wf_data['planner_team'] = Team(**wf_data['planner_team'])
                            if 'solver_team' in wf_data and wf_data['solver_team']:
                                wf_data['solver_team']['members'] = [ModelConfig(**m) for m in wf_data['solver_team']['members']]
                                wf_data['solver_team'] = Team(**wf_data['solver_team'])
                            if 'patcher_team' in wf_data and wf_data['patcher_team']:
                                wf_data['patcher_team']['members'] = [ModelConfig(**m) for m in wf_data['patcher_team']['members']]
                                wf_data['patcher_team'] = Team(**wf_data['patcher_team'])
                            if 'assembler_team' in wf_data and wf_data['assembler_team']:
                                wf_data['assembler_team']['members'] = [ModelConfig(**m) for m in wf_data['assembler_team']['members']]
                                wf_data['assembler_team'] = Team(**wf_data['assembler_team'])

                            # Reconstruct GauntletDefinition
                            if 'sub_problem_red_gauntlet' in wf_data and wf_data['sub_problem_red_gauntlet']:
                                wf_data['sub_problem_red_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['sub_problem_red_gauntlet']['rounds']]
                                wf_data['sub_problem_red_gauntlet'] = GauntletDefinition(**wf_data['sub_problem_red_gauntlet'])
                            if 'sub_problem_gold_gauntlet' in wf_data and wf_data['sub_problem_gold_gauntlet']:
                                wf_data['sub_problem_gold_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['sub_problem_gold_gauntlet']['rounds']]
                                wf_data['sub_problem_gold_gauntlet'] = GauntletDefinition(**wf_data['sub_problem_gold_gauntlet'])
                            if 'final_red_gauntlet' in wf_data and wf_data['final_red_gauntlet']:
                                wf_data['final_red_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['final_red_gauntlet']['rounds']]
                                wf_data['final_red_gauntlet'] = GauntletDefinition(**wf_data['final_red_gauntlet'])
                            if 'final_gold_gauntlet' in wf_data and wf_data['final_gold_gauntlet']:
                                wf_data['final_gold_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['final_gold_gauntlet']['rounds']]
                                wf_data['final_gold_gauntlet'] = GauntletDefinition(**wf_data['final_gold_gauntlet'])
                            if 'solver_generation_gauntlet' in wf_data and wf_data['solver_generation_gauntlet']:
                                wf_data['solver_generation_gauntlet']['rounds'] = [GauntletRoundRule(**r) for r in wf_data['solver_generation_gauntlet']['rounds']]
                                wf_data['solver_generation_gauntlet'] = GauntletDefinition(**wf_data['solver_generation_gauntlet'])

                            # Reconstruct DecompositionPlan
                            if 'decomposition_plan' in wf_data and wf_data['decomposition_plan']:
                                if 'sub_problems' in wf_data['decomposition_plan'] and wf_data['decomposition_plan']['sub_problems']:
                                    wf_data['decomposition_plan']['sub_problems'] = [SubProblem(**sp) for sp in wf_data['decomposition_plan']['sub_problems']]
                                wf_data['decomposition_plan'] = DecompositionPlan(**wf_data['decomposition_plan'])

                            # Reconstruct SolutionAttempt
                            if 'sub_problem_solutions' in wf_data and wf_data['sub_problem_solutions']:
                                wf_data['sub_problem_solutions'] = {k: SolutionAttempt(**v) for k, v in wf_data['sub_problem_solutions'].items()}
                            if 'final_solution' in wf_data and wf_data['final_solution']:
                                wf_data['final_solution'] = SolutionAttempt(**wf_data['final_solution'])

                            # Reconstruct CritiqueReport and VerificationReport
                            if 'all_critique_reports' in wf_data and wf_data['all_critique_reports']:
                                wf_data['all_critique_reports'] = [CritiqueReport(**cr) for cr in wf_data['all_critique_reports']]
                            if 'all_verification_reports' in wf_data and wf_data['all_verification_reports']:
                                wf_data['all_verification_reports'] = [VerificationReport(**vr) for vr in wf_data['all_verification_reports']]

                            # Reconstruct WorkflowState
                            self.history[wf_id] = WorkflowState(**wf_data)
                        except Exception as e:
                            print(f"Error reconstructing workflow {wf_id}: {e}. Skipping this entry.")
                    print(f"Loaded {len(self.history)} workflow history entries.")
                except json.JSONDecodeError:
                    print(f"Error decoding workflow history file: {self.history_file}. Starting with empty history.")
                    self.history = {}
        else:
            self.history = {}

    def _save_history(self) -> None:
        """
        Saves the current workflow history to the JSON file.
        """
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, cls=EnhancedJSONEncoder)

    def add_workflow_to_history(self, workflow_state: WorkflowState) -> None:
        """
        Adds a completed, failed, or cancelled workflow to the history.
        """
        self.history[workflow_state.workflow_id] = workflow_state
        self._save_history()

    def get_all_historical_workflows(self) -> List[WorkflowState]:
        """
        Retrieves all historical workflow states.
        """
        return list(self.history.values())

    def get_historical_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Retrieves a specific historical workflow state by ID.
        """
        return self.history.get(workflow_id)

    def clear_history(self) -> None:
        """
        Clears all workflow history.
        """
        self.history = {}
        self._save_history()
