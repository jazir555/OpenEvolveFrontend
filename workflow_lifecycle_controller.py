"""
OpenEvolve Workflow Lifecycle Controller

This module provides comprehensive lifecycle controls for OpenEvolve workflows
within the BubbleLabs interface, including start, pause, resume, stop, cancel,
and restart functionality.
"""

from ui_shim import ui as st
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from workflow_structures import WorkflowState
from openevolve_bubblelabs_api import openevolve_bubblelabs_integration


class WorkflowLifecycleController:
    """
    Provides comprehensive lifecycle controls for OpenEvolve workflows.
    """
    
    def __init__(self):
        self.integration = openevolve_bubblelabs_integration
    
    def show_workflow_lifecycle_controls(self):
        """
        Show comprehensive workflow lifecycle controls in the UI.
        """
        st.header("🔄 OpenEvolve Workflow Lifecycle Management")
        st.markdown("""
        Complete control over OpenEvolve workflow execution lifecycle:
        - **Start**: Initialize and begin workflow execution
        - **Pause**: Temporarily suspend workflow execution
        - **Resume**: Continue execution from where paused
        - **Stop**: Gracefully terminate workflow execution
        - **Cancel**: Immediately terminate workflow execution
        - **Restart**: Create and start a new workflow instance
        """)
        
        # Workflow selection
        st.subheader("Workflow Selection")
        instances = self.integration.list_workflow_instances()
        
        if not instances:
            st.info("No workflow instances available. Create a workflow instance first.")
            return
        
        # Create a mapping of readable names to instance IDs
        instance_options = {}
        for instance in instances:
            readable_name = f"{instance['instance_id'][:8]} - {instance['status']}"
            instance_options[readable_name] = instance['instance_id']
        
        if instance_options:
            selected_readable = st.selectbox(
                "Select Workflow Instance",
                options=list(instance_options.keys()),
                format_func=lambda x: x
            )
            selected_instance_id = instance_options[selected_readable]
            
            # Show current status
            status_info = self.integration.get_workflow_instance_status(selected_instance_id)
            if "error" in status_info:
                st.error(f"Error getting workflow status: {status_info['error']}")
                return
            
            # Display current status
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", f"{self._get_status_icon(status_info['status'])} {status_info['status'].upper()}")
            with col2:
                st.metric("Current Stage", status_info['current_stage'])
            with col3:
                st.metric("Progress", f"{status_info['progress'] * 100:.1f}%")
            with col4:
                if status_info['start_time']:
                    duration = time.time() - status_info['start_time']
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    st.metric("Duration", f"{minutes}m {seconds}s")
        
            # Control buttons arranged for easy access
            self._render_control_buttons(selected_instance_id, status_info['status'])
            
            # Instance details
            self._show_instance_details(selected_instance_id)
        else:
            st.info("No workflow instances available.")
    
    def _render_control_buttons(self, instance_id: str, current_status: str):
        """
        Render the control buttons based on current status.
        """
        st.subheader("Workflow Controls")
        
        # Create columns for control buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Determine which buttons should be active based on current status
        can_start = current_status in ["created", "pending", "stopped", "cancelled", "failed"]
        can_pause = current_status == "running"
        can_resume = current_status == "paused"
        can_stop = current_status in ["running", "pending"]
        can_cancel = current_status in ["running", "pending", "paused"]
        can_restart = current_status in ["completed", "failed", "cancelled", "stopped"]
        
        with col1:
            if st.button("▶️ Start", disabled=not can_start, key=f"start_{instance_id}"):
                result = self.integration.start_workflow_instance(instance_id)
                if "error" in result:
                    st.error(f"Start failed: {result['error']}")
                else:
                    st.success(f"Workflow started: {result['message']}")
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause", disabled=not can_pause, key=f"pause_{instance_id}"):
                result = self.integration.pause_workflow_instance(instance_id)
                if "error" in result:
                    st.error(f"Pause failed: {result['error']}")
                else:
                    st.success(f"Workflow paused: {result['message']}")
                st.rerun()
        
        with col3:
            if st.button("▶️ Resume", disabled=not can_resume, key=f"resume_{instance_id}"):
                result = self.integration.resume_workflow_instance(instance_id)
                if "error" in result:
                    st.error(f"Resume failed: {result['error']}")
                else:
                    st.success(f"Workflow resumed: {result['message']}")
                st.rerun()
        
        with col4:
            if st.button("⏹️ Stop", disabled=not can_stop, key=f"stop_{instance_id}"):
                result = self.integration.stop_workflow_instance(instance_id)
                if "error" in result:
                    st.error(f"Stop failed: {result['error']}")
                else:
                    st.success(f"Workflow stopped: {result['message']}")
                st.rerun()
        
        with col5:
            if st.button("🚫 Cancel", disabled=not can_cancel, key=f"cancel_{instance_id}"):
                result = self.integration.cancel_workflow_instance(instance_id)
                if "error" in result:
                    st.error(f"Cancel failed: {result['error']}")
                else:
                    st.success(f"Workflow cancelled: {result['message']}")
                st.rerun()
        
        with col6:
            if st.button("🔁 Restart", disabled=not can_restart, key=f"restart_{instance_id}"):
                result = self.integration.restart_workflow_instance(instance_id)
                if "error" in result:
                    st.error(f"Restart failed: {result['error']}")
                else:
                    st.success(f"Workflow restarted: {result['message']}")
                    st.info(f"New instance ID: {result['new_instance_id']}")
                st.rerun()
    
    def _show_instance_details(self, instance_id: str):
        """
        Show detailed information about the selected instance.
        """
        st.subheader("Instance Details")
        
        status_info = self.integration.get_workflow_instance_status(instance_id)
        if "error" in status_info:
            st.error(f"Error getting workflow status: {status_info['error']}")
            return
        
        # Create tabs for different details
        details_tabs = st.tabs(["Status", "Parameters", "Timeline", "Errors"])
        
        with details_tabs[0]:
            st.json({
                "instance_id": status_info["instance_id"],
                "status": status_info["status"],
                "current_stage": status_info["current_stage"],
                "progress": f"{status_info['progress'] * 100:.2f}%",
                "start_time": datetime.fromtimestamp(status_info["start_time"]).isoformat() if status_info["start_time"] else "N/A",
                "end_time": datetime.fromtimestamp(status_info["end_time"]).isoformat() if status_info["end_time"] else "N/A",
                "execution_time": f"{status_info['execution_time']:.2f}s" if status_info["execution_time"] else "N/A"
            })
        
        with details_tabs[1]:
            # Get the actual workflow state to show parameters
            workflow_state = self.integration.workflow_instances.get(instance_id)
            if workflow_state:
                # Only show relevant parameters
                params = {}
                for attr_name in dir(workflow_state):
                    if not attr_name.startswith('_') and not callable(getattr(workflow_state, attr_name)):
                        attr_value = getattr(workflow_state, attr_name)
                        if isinstance(attr_value, (str, int, float, bool, list, dict)) and len(str(attr_value)) < 1000:
                            params[attr_name] = attr_value
                st.json(params)
        
        with details_tabs[2]:
            # Show workflow timeline if available
            if status_info["start_time"]:
                timeline_data = [
                    {"event": "Created", "time": status_info["start_time"]},
                    {"event": "Started", "time": status_info["start_time"]},  # Simplified for demo
                    {"event": "Completed", "time": status_info["end_time"]} if status_info["end_time"] else {"event": "In Progress", "time": time.time()}
                ]
                
                for item in timeline_data:
                    event_time = datetime.fromtimestamp(item["time"]).strftime('%Y-%m-%d %H:%M:%S')
                    st.write(f"**{item['event']}**: {event_time}")
        
        with details_tabs[3]:
            if status_info.get("error_message"):
                st.error(f"Error: {status_info['error_message']}")
            else:
                st.success("No errors reported for this instance.")
    
    def _get_status_icon(self, status: str) -> str:
        """
        Get appropriate icon for workflow status.
        """
        status_icons = {
            'created': '🆕',
            'pending': '⏳',
            'running': '🏃',
            'paused': '⏸️',
            'stopping': '🛑',
            'stopped': '⏹️',
            'completed': '[OK]',
            'failed': '[FAIL]',
            'cancelled': '🚫'
        }
        return status_icons.get(status.lower(), '❓')
    
    def create_new_workflow(self):
        """
        UI for creating a new workflow.
        """
        st.subheader("Create New Workflow")
        
        # Workflow type selection
        workflow_type = st.selectbox(
            "Workflow Type",
            options=["evolution", "adversarial", "sovereign"],
            format_func=lambda x: x.title()
        )
        
        # Basic workflow parameters
        col1, col2 = st.columns(2)
        
        with col1:
            workflow_name = st.text_input("Workflow Name", value=f"{workflow_type.title()} Workflow {int(time.time())}")
        
        with col2:
            description = st.text_input("Description", value=f"OpenEvolve {workflow_type} workflow")
        
        # Problem statement
        problem_statement = st.text_area(
            "Problem Statement",
            placeholder="Enter the problem to solve with OpenEvolve...",
            height=150
        )
        
        if st.button("Create Workflow Definition"):
            if not problem_statement.strip():
                st.error("Please enter a problem statement")
                return
            
            # Create initial parameters
            parameters = {
                "max_iterations": 100,
                "population_size": 50,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 4096,
                "num_islands": 5,
                "migration_rate": 0.1,
                "archive_size": 100,
                "enable_qd_evolution": False,
                "enable_multi_objective": False,
                "enable_adversarial": False,
                "memory_limit_mb": 2048,
                "cpu_limit": 1.0
            }
            
            # Create the workflow definition
            definition_id = self.integration.create_workflow_definition(
                name=workflow_name,
                description=description,
                workflow_type=workflow_type,
                parameters=parameters
            )
            
            st.success(f"Workflow definition created successfully! ID: {definition_id}")
            
            # Store in session state for later use
            if "created_workflow_defs" not in st.session_state:
                st.session_state.created_workflow_defs = []
            st.session_state.created_workflow_defs.append({
                "id": definition_id,
                "name": workflow_name,
                "type": workflow_type,
                "created_at": time.time()
            })
    
    def create_and_run_instance(self):
        """
        UI for creating and running a workflow instance from a definition.
        """
        st.subheader("Create and Run Workflow Instance")
        
        # Get available workflow definitions
        defs = self.integration.list_workflow_definitions()
        
        if not defs:
            st.info("No workflow definitions available. Create a definition first.")
            return
        
        # Create selection
        def_options = {f"{d['name']} ({d['workflow_type']})": d['id'] for d in defs}
        selected_def_name = st.selectbox(
            "Select Workflow Definition",
            options=list(def_options.keys()),
            format_func=lambda x: x
        )
        selected_def_id = def_options[selected_def_name]
        
        # Instance inputs
        inputs = st.text_area(
            "Input Parameters (JSON)",
            value='{"content": "Enter your content here", "problem_statement": "Enter problem to solve"}',
            height=150
        )
        
        try:
            input_dict = json.loads(inputs)
        except json.JSONDecodeError:
            st.error("Invalid JSON in input parameters")
            return
        
        if st.button("Create and Run Instance"):
            # Create the workflow instance
            instance_id = self.integration.create_workflow_instance(
                definition_id=selected_def_id,
                instance_name=f"Instance-{int(time.time())}",
                inputs=input_dict
            )
            
            # Start the workflow
            result = self.integration.start_workflow_instance(instance_id)
            
            if "error" in result:
                st.error(f"Failed to start workflow: {result['error']}")
            else:
                st.success(f"Workflow instance created and started! Instance ID: {instance_id}")
    
    def render_complete_lifecycle_ui(self):
        """
        Render the complete lifecycle management UI.
        """
        # Create tabs for different functions
        tabs = st.tabs(["Workflow Controls", "Create Workflow", "Create Instance"])
        
        with tabs[0]:
            self.show_workflow_lifecycle_controls()
        
        with tabs[1]:
            self.create_new_workflow()
        
        with tabs[2]:
            self.create_and_run_instance()


# Global function to render lifecycle controls
def render_workflow_lifecycle_controls():
    """
    Global function to render the workflow lifecycle controls.
    """
    controller = WorkflowLifecycleController()
    controller.render_complete_lifecycle_ui()
