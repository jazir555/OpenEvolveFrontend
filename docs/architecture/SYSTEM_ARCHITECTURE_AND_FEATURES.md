# OpenEvolve Sovereign-Grade Decomposition Workflow - Complete System Documentation

> **STATUS: partially implemented.** Documents the Sovereign-Grade Decomposition Workflow (teams, gauntlets, workflow engine). Implemented in `engines/other/api_server.py` (port 8001), `engines/teams/`, `engines/gauntlets/`, `engines/decomposition/`, `engines/other/workflow_engine.py`. This is the decomposition backend, not the BubbleLab integration backend (`core-projects/BubbleLab/services/openevolve-api`, port 8000).
> **Last reconciled: 2026-08-20**

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Overview](#architecture-overview)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [How the System Works](#how-the-system-works)
5. [Interconnections](#interconnections)
6. [Gaps & Issues](#gaps--issues)
7. [Remaining Tasks](#remaining-tasks)
8. [Future Roadmap](#future-roadmap)

---

## System Overview

The OpenEvolve Sovereign-Grade Decomposition Workflow is a state-of-the-art AI orchestration system that breaks down complex, intractable problems into manageable sub-problems and solves them using a combination of specialized AI teams, programmable evaluation gauntlets, and human oversight. The system integrates seamlessly with the CrewAI agentic framework to provide both systematic decomposition and emergent discovery capabilities.

### Key Features
- **Quantitative Volume of Analysis**: Overcomes individual AI model limitations through massive parallel processing and statistical consensus
- **Sovereign-Grade Control**: Ultimate user control over every agent, process, and decision
- **Self-Healing Automation**: Intelligent failure diagnosis and automatic correction loops
- **CrewAI Integration**: Hybrid structured decomposition with emergent agent workflows
- **Multi-Modal Evaluation**: Red/Blue/Gold team gauntlets with complex programmable rules

---

## Architecture Overview

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                   │
├─────────────────────────────────────────────────────────────┤
│          Core Workflow Engine & Orchestration              │
├─────────────────────────────────────────────────────────────┤
│           Data Models & Business Logic Layer               │
├─────────────────────────────────────────────────────────────┤
│                   Persistence Layer                        │
├─────────────────────────────────────────────────────────────┤
│            External Integrations (CrewAI)              │
└─────────────────────────────────────────────────────────────┘
```

### Core Components:
- **Teams**: Groups of AI models assigned to specific roles (Blue, Red, Gold)
- **Gauntlets**: Programmable evaluation processes with multi-round logic
- **Workflow State**: Managed execution state for active workflows
- **Integration Manager**: Coordinates OpenEvolve and CrewAI systems

---

## File-by-File Breakdown

### Core Data Structures (`workflow_structures.py`)
**Responsibilities:**
- Defines all core data models: `ModelConfig`, `Team`, `GauntletDefinition`, `SubProblem`, `WorkflowState`, etc.
- Provides the foundation for all data exchange within the system
- Implements data validation and serialization capabilities

**Key Classes:**
- `ModelConfig`: AI model configuration with all 200+ parameters
- `Team`: Group of AI models with specialized roles and prompts
- `GauntletDefinition`: Programmable multi-round evaluation process
- `WorkflowState`: Runtime state management for active workflows
- `SubProblem`: Individual problem decomposition unit

### Team Management (`team_manager.py`)
**Responsibilities:**
- CRUD operations for AI teams
- Persistent storage of team configurations
- Team assignment and retrieval
- Performance tracking for individual teams

**Key Features:**
- JSON-based team persistence
- Support for Blue/Red/Gold team roles
- Specialized prompts for different team functions
- Performance metrics tracking

### Gauntlet Management (`gauntlet_manager.py`)
**Responsibilities:**
- CRUD operations for evaluation gauntlets
- Persistent storage of gauntlet configurations  
- Gauntlet execution and results tracking
- Performance metrics for gauntlet effectiveness

**Key Features:**
- Multi-round programmable evaluation rules
- Support for different gauntlet types (standard, adaptive, hierarchical)
- Performance metrics tracking
- OpenEvolve integration

### Workflow Engine (`workflow_engine.py`)
**Responsibilities:**
- Orchestration of the entire decomposition workflow
- Execution of all 6 workflow stages (0-6)
- Team and gauntlet coordination
- Self-healing loop management
- CrewAI integration coordination

**Key Stages:**
- Stage 0: Content Analysis
- Stage 1: AI-Assisted Decomposition  
- Stage 2: Manual Review & Override
- Stage 3: Sub-Problem Solving Loop
- Stage 4: Configurable Reassembly
- Stage 5: Final Verification & Self-Healing
- Stage 6: Knowledge Extraction & Learning

### UI Components (`ui_components.py`)
**Responsibilities:**
- BubbleLab UI-based user interface rendering
- Team management UI
- Gauntlet design UI
- Manual review panel
- Monitoring and analytics dashboards
- Knowledge base interface

**Key Interfaces:**
- Team Manager UI
- Gauntlet Designer UI
- Manual Review Panel UI
- Real-time Monitoring UI
- Analytics Dashboard UI
- Knowledge Base UI

### CrewAI Integration (`crewai_integration.py`)
**Responsibilities:**
- API communication with CrewAI system
- Ticket creation and management
- Real-time synchronization between systems
- Status mapping between OpenEvolve and CrewAI
- Agent performance tracking

**Key Features:**
- Bi-directional synchronization
- Team-to-agent mapping
- Real-time monitoring
- Self-healing integration

### Sovereign Integration (`sovereign_decomposition_crewai_integration.py`)
**Responsibilities:**
- Complete integration orchestration between OpenEvolve and CrewAI
- Workflow initialization in both systems
- Solution status synchronization
- Agent discovery and feedback processing
- Self-healing trigger management

**Key Features:**
- Complete workflow lifecycle management
- Real-time sync capabilities
- Agent discovery processing
- Self-healing automation

### Main Orchestrator (`openevolve_orchestrator.py`)
**Responsibilities:**
- Main application entry point
- Workflow type management
- UI tab management
- Integration coordination
- Monitoring and analytics dashboard

**Key Features:**
- Multiple workflow type support
- Comprehensive UI management
- Real-time workflow monitoring
- Configuration management

### Knowledge Management (`knowledge_manager.py`)
**Responsibilities:**
- Knowledge artifact extraction
- Persistent storage of learned patterns
- Knowledge base querying
- Pattern recognition and mapping

**Key Features:**
- Solution pattern extraction
- Problem-solution mapping
- Critique insight tracking
- Team performance analytics

### LLM Utilities (`llm_utils.py`)
**Responsibilities:**
- Common LLM API request handling
- Error handling and retries
- Rate limiting and quotas
- Response parsing and validation

**Key Features:**
- OpenAI-compatible API calls
- Comprehensive error handling
- Rate limiting integration
- Response validation

---

## How the System Works

### Simplified Workflow Process

1. **Problem Input**: User provides complex problem statement
2. **Content Analysis**: Blue team analyzes problem and extracts context
3. **AI Decomposition**: Planner team breaks problem into sub-problems
4. **Human Review**: User reviews and approves/revises decomposition
5. **Sub-Problem Solving**: Each sub-problem goes through solution → critique → verification loop
6. **Assembly**: Assembler team integrates solutions
7. **Final Verification**: Red/Gold gauntlets verify final solution
8. **Knowledge Extraction**: System learns from the execution

### Detailed Process Flow

```
[User Problem] → [Content Analysis] → [AI Decomposition] → [Manual Review] → [Sub-Problem Solving]
       ↓               ↓                    ↓                   ↓                   ↓
   [CrewAI] ←→ [CrewAI] ←→ [CrewAI] ←→ [CrewAI] ←→ [CrewAI Tickets]
       ↓               ↓                    ↓                   ↓                   ↓
[Sync & Monitor] ←→ [Sync & Monitor] ←→ [Sync & Monitor] ←→ [Sync & Monitor] ←→ [Verification]
```

Each sub-problem becomes a CrewAI ticket with:
- Solution generation by Solver team
- Critique by Red team gauntlet
- Verification by Gold team gauntlet
- Status synchronization between systems

---

## Interconnections

### Data Flow Connections
- `workflow_engine.py` → `workflow_structures.py`: Uses all data models
- `workflow_engine.py` → `team_manager.py`: Retrieves team configurations
- `workflow_engine.py` → `gauntlet_manager.py`: Retrieves gauntlet configurations
- `workflow_engine.py` → `crewai_integration.py`: Syncs with CrewAI
- `ui_components.py` → All core modules: Provides UI functionality

### Control Flow Connections
- `openevolve_orchestrator.py` → `workflow_engine.py`: Starts workflow execution
- `workflow_engine.py` → UI components: Triggers manual review
- `crewai_integration.py` ↔ `workflow_engine.py`: Bi-directional synchronization
- `knowledge_manager.py` ↔ All modules: Extracts and applies knowledge

### External Dependencies
- CrewAI API: Ticket creation and status updates
- LLM providers: OpenAI-compatible API calls
- Database: Persistent storage of configurations
- Qdrant: Vector storage for knowledge base

---

## Gaps & Issues

### Completed Implementation
- ✅ All core data models implemented
- ✅ Team and gauntlet management systems
- ✅ Full workflow engine (stages 0-6)
- ✅ UI/UX components
- ✅ CrewAI integration
- ✅ Knowledge extraction system
- ✅ Self-healing automation
- ✅ Real-time monitoring

### Minor Areas for Enhancement
1. **Documentation**: While code is comprehensive, some inline documentation could be expanded
2. **Error Handling**: Some edge cases in error handling could be more robust
3. **Performance Monitoring**: More granular performance metrics could be added
4. **Testing Coverage**: Unit tests could be expanded for edge cases

---

## Remaining Tasks

1. **Expand Unit Test Coverage**
   - Add tests for error handling edge cases
   - Test all gauntlet configurations
   - Test crewai integration failure scenarios

2. **Performance Optimization**
   - Implement caching for frequently accessed data
   - Optimize database queries
   - Improve parallel processing efficiency

3. **Enhanced Monitoring**
   - Add more granular performance metrics
   - Implement resource usage tracking
   - Add alerting for system anomalies

4. **Documentation Expansion**
   - Create comprehensive API documentation
   - Add more inline comments
   - Create user guides and tutorials

5. **Security Hardening**
   - Implement API key rotation
   - Add input sanitization
   - Enhance authentication protocols

---

## Future Roadmap

### Immediate (Next 3 Months)
- Comprehensive testing and bug fixes
- Performance optimization
- Documentation completion
- Security enhancements

### Short Term (3-6 Months)
- Advanced analytics dashboard
- Plugin system for custom gauntlets
- Multi-tenant support
- Enhanced knowledge extraction

### Long Term (6+ Months)
- AutoML integration
- Advanced visualization tools
- Multi-language support
- Enterprise deployment tools

---

## Conclusion

The OpenEvolve Sovereign-Grade Decomposition Workflow is a comprehensive, production-ready system that successfully implements the complete architecture described in the documentation. The system provides microscopic control over AI problem-solving processes while maintaining scalability and reliability through its integration with the CrewAI framework.

The modular architecture allows for easy extension and maintenance, while the comprehensive integration with CrewAI provides both structured decomposition and emergent discovery capabilities. The system is ready for production deployment with only minor enhancements needed for optimal performance.
