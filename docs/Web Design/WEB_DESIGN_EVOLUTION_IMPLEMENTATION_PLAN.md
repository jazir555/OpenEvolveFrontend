# Web Design Evolution Platform - Comprehensive Implementation Plan

## Executive Summary

This document outlines the complete technical implementation strategy for building an evolutionary web design platform that uses visual LLM judges to evaluate and optimize web designs through genetic algorithms. The platform is built on **BubbleLab** (workflow automation foundation) and **OpenEvolve** (evolutionary computation engine), enabling users to generate 1,000+ design variations, visualize their evolution through a mitosis-style animation, and export scientifically optimized designs.

**Platform Foundation:**
- 🟢 **BubbleLab**: React 19, Vite, Hono, Drizzle ORM, Clerk auth, @xyflow/react visualization
- 🔧 **OpenEvolve**: Mutation engine, decomposition workflow, MCTS/MDAP optimization, adversarial/team/gauntlet systems, parameter manager, maker automation

**Development Impact:**
- Task counts tracked in `docs/todos/WEB_DESIGN_EVOLUTION_TODO_LIST.md`
- BubbleLab/OpenEvolve reduce new build effort by ~47% (synced with todo list)
- Time savings estimate: TBD after final breakdown

---

## Table of Contents

1. [Vision & Core Value Proposition](#1-vision--core-value-proposition)
   - 1.1 [Problem Statement](#11-problem-statement)
   - 1.2 [Solution](#12-solution)
   - 1.3 [Key Differentiators](#13-key-differentiators)

2. [Technical Architecture](#2-technical-architecture)
   - 2.1 [High-Level Architecture (BubbleLab-Based)](#21-high-level-architecture-bubblelab-based)
   - 2.2 [BubbleLab Integration Strategy](#22-bubblelab-integration-strategy)
     - 2.2.1 [Overview](#221-overview)
     - 2.2.2 [Component Mapping](#222-component-mapping)
     - 2.2.3 [OpenEvolve Integration](#223-openevolve-integration)
   - 2.3 [Technology Stack (BubbleLab-Based)](#23-technology-stack-bubblelab-based)
     - 2.3.1 [Frontend Stack (Extended from BubbleLab)](#231-frontend-stack-extended-from-bubblelab)
     - 2.3.2 [Backend Stack (Extended from BubbleLab)](#232-backend-stack-extended-from-bubblelab)
     - 2.3.3 [Infrastructure (Extended from BubbleLab)](#233-infrastructure-extended-from-bubblelab)
     - 2.3.4 [AI/ML Services (Extended from BubbleLab)](#234-aiml-services-extended-from-bubblelab)
     - 2.3.5 [OpenEvolve Python Integration](#235-openevolve-python-integration)
   - 2.4 [Week 1 Setup: BubbleLab + OpenEvolve Integration](#24-week-1-setup-bubblelab--openevolve-integration)
     - 2.4.1 [Development Environment Prerequisites](#241-development-environment-prerequisites)
     - 2.4.2 [Day 1 Setup Tasks](#242-day-1-setup-tasks)
     - 2.4.3 [Integration Architecture (TypeScript → Python)](#243-integration-architecture-typescript--python)
     - 2.4.4 [Week 1 Task Breakdown](#244-week-1-task-breakdown)

3. [System Components](#3-system-components)
   - 3.1 [Screenshot Renderer Service](#31-screenshot-renderer-service)
   - 3.2 [Visual LLM Judge Service](#32-visual-llm-judge-service)
   - 3.3 [Mutation Engine (OpenEvolve Integration 🔧)](#33-mutation-engine-openevolve-integration-)
     - 3.3.1 [Color Mutations](#331-color-mutations)
     - 3.3.2 [Typography Mutations](#332-typography-mutations)
     - 3.3.3 [Layout Mutations](#333-layout-mutations)
     - 3.3.4 [Content Mutations](#334-content-mutations)
     - 3.3.5 [Component Mutations](#335-component-mutations)
   - 3.4 [Evolution Orchestrator](#34-evolution-orchestrator)
   - 3.5 [Event Bus & Real-time Communication](#35-event-bus--real-time-communication)
   - 3.6 [Adversarial & Team System (OpenEvolve Integration 🔧)](#36-adversarial--team-system-openevolve-integration-)
   - 3.7 [Decomposition Workflow Engine (OpenEvolve Integration 🔧)](#37-decomposition-workflow-engine-openevolve-integration-)
   - 3.8 [Gauntlet & Gold Team Verification (OpenEvolve Integration 🔧)](#38-gauntlet--gold-team-verification-openevolve-integration-)

4. [Data Models & Schemas](#4-data-models--schemas)
   - 4.1 [Core Entities](#41-core-entities)
     - 4.1.1 [Design](#411-design)
     - 4.1.2 [Fitness Criteria](#412-fitness-criteria)
     - 4.1.3 [Judge Score](#413-judge-score)
     - 4.1.4 [Evolution Request](#414-evolution-request)
     - 4.1.5 [Evolution Result](#415-evolution-result)
   - 4.2 [Database Schema](#42-database-schema)

5. [API Design](#5-api-design)
   - 5.1 [REST API Endpoints](#51-rest-api-endpoints)
   - 5.2 [WebSocket Events](#52-websocket-events)
   - 5.3 [GraphQL Schema (Optional)](#53-graphql-schema-optional)

6. [Visual LLM Integration](#6-visual-llm-integration)
   - 6.1 [Judge Agent Configuration](#61-judge-agent-configuration)
   - 6.2 [Aggregation Strategy](#62-aggregation-strategy)
   - 6.3 [Cost Optimization Strategy](#63-cost-optimization-strategy)

7. [Mutation Engine Design (OpenEvolve Integration 🔧)](#7-mutation-engine-design-openevolve-integration-)
   - 7.1 [Mutation Strategy Matrix (from OpenEvolve)](#71-mutation-strategy-matrix-from-openevolve)
   - 7.2 [Adaptive Mutations](#72-adaptive-mutations)
   - 7.3 [Constraint-Aware Mutations](#73-constraint-aware-mutations)

8. [Evolution Orchestrator](#8-evolution-orchestrator)
   - 8.1 [Pipeline Architecture](#81-pipeline-architecture)
   - 8.2 [Checkpoint & Resume](#82-checkpoint--resume)
   - 8.3 [Progress Tracking](#83-progress-tracking)
   - 8.4 [Adversarial & Team Phases](#84-adversarial--team-phases)
   - 8.5 [Gauntlet & Gold Team Gates](#85-gauntlet--gold-team-gates)
   - 8.6 [Decomposition Preprocessing](#86-decomposition-preprocessing)

9. [Frontend & Visualization](#9-frontend--visualization)
   - 9.1 [Mitosis Animation Architecture](#91-mitosis-animation-architecture)
   - 9.2 [Interactive Features](#92-interactive-features)
   - 9.3 [Timeline Scrubber](#93-timeline-scrubber)
   - 9.4 [UI Components](#94-ui-components)

10. [Infrastructure & DevOps](#10-infrastructure--devops)
    - 10.1 [Container Architecture (Hybrid: TypeScript + Python)](#101-container-architecture-hybrid-typescript--python)
    - 10.2 [Kubernetes Configuration (Production)](#102-kubernetes-configuration-production)
    - 10.3 [CI/CD Pipeline](#103-cicd-pipeline)

11. [Security & Privacy](#11-security--privacy)
    - 11.1 [Authentication & Authorization](#111-authentication--authorization)
    - 11.2 [Data Privacy](#112-data-privacy)
    - 11.3 [API Security](#113-api-security)

12. [Performance Optimization](#12-performance-optimization)
    - 12.1 [Screenshot Caching](#121-screenshot-caching)
    - 12.2 [Parallel Processing](#122-parallel-processing)
    - 12.3 [Database Optimization](#123-database-optimization)

13. [Testing Strategy](#13-testing-strategy)
    - 13.1 [Unit Testing](#131-unit-testing)
    - 13.2 [Integration Testing](#132-integration-testing)
    - 13.3 [E2E Testing](#133-e2e-testing)

14. [Deployment Strategy](#14-deployment-strategy)
    - 14.1 [Phased Rollout](#141-phased-rollout)
    - 14.2 [Feature Flags](#142-feature-flags)

15. [Monitoring & Observability](#15-monitoring--observability)
    - 15.1 [Metrics](#151-metrics)
    - 15.2 [Logging](#152-logging)
    - 15.3 [Alerting](#153-alerting)

16. [Scaling Strategy](#16-scaling-strategy)
    - 16.1 [Horizontal Scaling](#161-horizontal-scaling)
    - 16.2 [Vertical Scaling](#162-vertical-scaling)

17. [Cost Management](#17-cost-management)
    - 17.1 [Cost Breakdown (Per Evolution)](#171-cost-breakdown-per-evolution)
    - 17.2 [Pricing Strategy](#172-pricing-strategy)
    - 17.3 [Cost Optimization](#173-cost-optimization)

18. [Risk Mitigation](#18-risk-mitigation)
    - 18.1 [Technical Risks](#181-technical-risks)
    - 18.2 [Business Risks](#182-business-risks)
    - 18.3 [Legal Risks](#183-legal-risks)
19. [Full Implementation Addenda](#19-full-implementation-addenda)
    - 19.1 [Credits, Billing & Tier Enforcement](#191-credits-billing--tier-enforcement)
    - 19.2 [Exports & Design System Assets](#192-exports--design-system-assets)
    - 19.3 [Feedback, Collaboration & Governance](#193-feedback-collaboration--governance)
    - 19.4 [Enterprise Readiness](#194-enterprise-readiness)
    - 19.5 [Operational Readiness](#195-operational-readiness)
20. [Production Readiness](#20-production-readiness)
    - 20.1 [Reliability & SLOs](#201-reliability--slos)
    - 20.2 [Security Hardening](#202-security-hardening)
    - 20.3 [Data Governance & Privacy](#203-data-governance--privacy)
    - 20.4 [Release Management](#204-release-management)
    - 20.5 [Support & Incident Response](#205-support--incident-response)
    - 20.6 [Compliance & Legal](#206-compliance--legal)

---

## 1. Vision & Core Value Proposition

### 1.1 Problem Statement

- Design iteration is slow (weeks to months)
- A/B testing is expensive ($15,000+ per test)
- Subjective feedback leads to endless revision cycles
- Current AI design tools are black boxes designers don't trust
- No scientific approach to design optimization

### 1.2 Solution

**Evolutionary Design Platform** that:
- Generates 1,000+ variations in <10 minutes
- Uses multi-agent visual LLM judges for fitness evaluation
- Provides explainable AI through evolution tree visualization
- Enables objective, criteria-driven design optimization
- Reduces design iteration from weeks to hours

### 1.3 Key Differentiators

1. **Visual Verification**: Every design decision is screenshot + LLM evaluated
2. **Explainable AI**: Full evolution tree with reasoning for each decision
3. **User-Defined Criteria**: Custom fitness functions for brand/goals
4. **Mitosis Animation**: Real-time visualization of evolution process
5. **Scientific Approach**: Survival of fittest based on objective metrics

---

## 2. Technical Architecture

### 2.1 High-Level Architecture (BubbleLab-Based)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BubbleLab Evolution Platform                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Frontend Layer (BubbleLab)                      │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ BubbleStudio UI  │  │ @xyflow/react   │  │ Zustand Stores  │   │  │
│  │  │ React 19.1.0    │  │ Evolution Trees │  │ State Management│   │  │
│  │  │ + Evolution     │  │ Mitosis Anim.   │  │ + Evolution     │   │  │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘   │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ TanStack Router │  │ Clerk Auth      │  │ Socket.io/Exec  │   │  │
│  │  │ Navigation      │  │ Authentication  │  │ Log Streaming   │   │  │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                  │
│                                    │ WebSocket/HTTP                   │
│                                    ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   API Gateway Layer (BubbleLab)                    │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ Hono Routes     │  │ Zod Validation  │  │ Error Handling  │   │  │
│  │  │ /api/evolution/*│  │ Schemas         │  │ Middleware      │   │  │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                  │
│                                    │ Service Calls                    │
│                                    ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   Application Layer (NEW + Extended)                │  │
│  │  ┌─────────────────────────────────────────────────────────┐     │  │
│  │  │           Evolution Orchestrator Service (NEW)           │     │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │     │  │
│  │  │  │ OpenEvolve   │  │ Selection    │  │ Fitness       │  │     │  │
│  │  │  │ Mutation     │  │ Algorithm    │  │ Aggregation    │  │     │  │
│  │  │  │ Engine 🔧    │  │ (NEW)        │  │ (NEW)          │  │     │  │
│  │  │  └──────────────┘  └──────────────┘  └────────────────┘  │     │  │
│  │  └─────────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                  │
│                                    │                                 │
│                                    ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Service Layer (Extended BubbleLab)              │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ Screenshot      │  │ AIAgentBubble   │  │ Storage         │   │  │
│  │  │ Renderer (NEW)   │  │ Extended Visual │  │ (S3/R2) ✅     │   │  │
│  │  │ Puppeteer       │  │ LLM Judges     │  │                 │   │  │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘   │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ Redis Cache     │  │ Drizzle ORM     │  │ Clerk Backend   │   │  │
│  │  │ (Extended)      │  │ Evolution ✅    │  │ Auth ✅         │   │  │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                  │
│                                    │                                 │
│                                    ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Data Layer (BubbleLab)                        │  │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │  │
│  │  │ PostgreSQL       │  │ Redis           │  │ S3/Cloudflare   │   │  │
│  │  │ Evolution ✅     │  │ Cache/Queue     │  │ R2 ✅           │   │  │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Legend: ✅ = Existing BubbleLab Component | NEW = New Component        │
│         🔧 = OpenEvolve Component | Extended = Modified Existing     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 BubbleLab Integration Strategy

#### 2.2.1 Overview

The Web Design Evolution Platform will be built as an extension of the **BubbleLab** workflow automation platform, leveraging its existing infrastructure, components, and architecture. Additionally, it integrates **OpenEvolve** for advanced mutation and decomposition engines.

BubbleLab provides a robust foundation with:

- **Frontend**: React 19 + Vite + TypeScript + Zustand + @xyflow/react + Tailwind CSS
- **Backend**: Hono + Bun + Drizzle ORM + PostgreSQL/SQLite support
- **Authentication**: Clerk for both frontend and backend
- **AI Integration**: Multi-provider LLM support (OpenAI, Anthropic, Google, DeepSeek)
- **Flow Visualization**: @xyflow/react-based workflow visualizer (adaptable for evolution trees)
- **State Management**: Zustand for application state
- **Real-time Updates**: Execution log streaming infrastructure

**OpenEvolve** provides evolutionary computation capabilities:

- **Mutation Engine**: Advanced genetic operators and mutation strategies
- **Decomposition Engine**: Problem decomposition, analysis, and workflow tools
- **Evolutionary Algorithms**: MCTS, MDAP, adversarial evolution
- **Team/Gauntlet System**: Red/blue/evaluator teams, gauntlet testing, gold team verification
- **Configuration & Adapters**: 272-parameter manager and adapter layer for orchestration
- **Maker Engine**: Automated code generation and optimization

#### 2.2.2 Component Mapping

| Evolution Platform Component | Source | Integration Strategy |
|----------------------------|---------------------|----------------------|
| Frontend UI | 🟢 BubbleStudio | Extend bubble-studio with evolution settings + insights pages (`BubbleLab/apps/bubble-studio/src/pages/EvolutionSettingsPage.tsx`, `BubbleLab/apps/bubble-studio/src/pages/EvolutionInsightsPage.tsx`, `/routes/evolution*.tsx`) |
| Flow Visualization | 🟢 FlowVisualizer (@xyflow/react) | Adapt FlowVisualizer for evolution tree visualization with mitosis animations |
| State Management | 🟢 Zustand stores | Create evolution-specific Zustand stores |
| Evolution UI Schemas | 🔧 OpenEvolve-Plugin | Schema mirrors wired into bubble-studio (`BubbleLab/apps/bubble-studio/src/lib/evolution/schemas.ts`, `BubbleLab/apps/bubble-studio/src/stores/evolutionSettingsStore.ts`, `BubbleLab/apps/bubble-studio/src/pages/EvolutionSettingsPage.tsx`) |
| Plugin Architecture | 🔧 OpenEvolve | Reuse bubblelabs_plugin_system.py for plugin lifecycle + event bus |
| API Routes | 🟢 Hono routes | Add new routes to bubblelab-api for evolution endpoints |
| Authentication | 🟢 Clerk | Reuse existing Clerk integration |
| Mutation Engine | 🔧 OpenEvolve | Integrate OpenEvolve mutation/decomposition engines |
| Decomposition Engine | 🔧 OpenEvolve | Use OpenEvolve problem decomposition and analysis + workflow |
| Evolutionary Algorithms | 🔧 OpenEvolve | MCTS, MDAP, adversarial evolution from OpenEvolve |
| Team/Gauntlet System | 🔧 OpenEvolve | Integrate red/blue/evaluator teams, gauntlet, gold team verification |
| Selection Algorithm | NEW | Implement tournament/roulette selection in TypeScript |
| Fitness Aggregation | NEW | Implement multi-criteria fitness aggregation |
| LLM Integration | 🟢 AIAgentBubble | Extend existing AIAgentBubble for multi-provider LLM calls with vision support |
| Database | 🟢 Drizzle ORM | Extend existing schema with evolution-specific tables |
| File Storage | 🟢 Storage service | Reuse existing S3/R2 storage integration |
| Real-time Updates | 🟢 Execution log streaming | Adapt execution log infrastructure for evolution events |
| Screenshot Rendering | NEW | Add Puppeteer service for HTML → Screenshot rendering |

#### 2.2.3 OpenEvolve Integration

**OpenEvolve** is an advanced evolutionary computation framework that provides the mutation and optimization engines for the platform. OpenEvolve components are marked with 🔧 in the architecture diagrams.

##### Key OpenEvolve Components:

1. **Mutation Engine** (`decomposition_engine.py`, `evolutionary_optimization.py`)
   - Genetic operators: crossover, mutation, selection
   - Adaptive mutation rates based on fitness landscape
   - Multi-objective optimization support
   - Decomposition-based problem solving

2. **Evolutionary Algorithms**
   - **MCTS Integration** (`mdap_engine.py`, `mdap_maker_mcts_unified.py`)
     - Monte Carlo Tree Search for solution space exploration
     - MDAP (Multi-Objective Decomposition Algorithm)
     - Maker-MCTS unified framework
   - **Adversarial Evolution** (`adversarial_mdap_mcts.py`, `adversarial_unified.py`)
     - Red-blue team testing for robust solutions
     - Adversarial mutation strategies
   - **Hybrid Systems** (`hybrid_maker_integration.py`)
     - Combined evolutionary and MCTS approaches
     - Multi-modal optimization

3. **Decomposition Engine + Workflow** (`decomposition_engine.py`, `decomposition_maker_integration.py`, `decomposition_crewai_bridge.py`)
   - Problem breakdown into sub-problems
   - Hierarchical optimization
   - Parallel evolution tracks

4. **Team + Gauntlet System** (`red_team.py`, `blue_team.py`, `evaluator_team.py`, `team_manager.py`, `gauntlet_manager.py`, `ragbits_integration/agents/gold_team_agent.py`)
   - Red/blue/evaluator assessments for robustness
   - Gauntlet-based stress testing and gating
   - Gold team verification for final winners

5. **Evolution Configuration + Adapters** (`evolution.py`, `parameter_manager.py`, `evolution_adapter.py`, `evolutionary_optimization.py`)
   - 272-parameter configuration surface
   - Adapter layer for orchestrator integration
   - Unified evolution workflow wrapper

6. **Maker Engine** (`maker_engine.py`, `mdap_maker_complete.py`)
   - Automated code generation
   - Solution synthesis
   - Pattern-based optimization

7. **Evolution UI Integration** (`bubblelabs_evolution_integration.py`, `bubblelabs_evolution_controls.py`)
   - BubbleLab UI dashboard for progress, metrics, and controls
   - BubbleLab web UI wired for settings + insights (`BubbleLab/apps/bubble-studio/src/pages/EvolutionSettingsPage.tsx`, `BubbleLab/apps/bubble-studio/src/pages/EvolutionInsightsPage.tsx`, `/stores/evolutionSettingsStore.ts`)
   - Continue porting advanced insights panels (adversarial/gauntlet/decomposition)

##### Integration Strategy:

```python
# Evolution Orchestrator will use OpenEvolve components
from openevolve.decomposition_engine import DecompositionEngine
from openevolve.evolutionary_optimization import EvolutionaryOptimizer
from openevolve.mdap_engine import MDAPEngine
from openevolve.maker_engine import MakerEngine

class EvolutionOrchestrator:
    def __init__(self):
        self.mutation_engine = EvolutionaryOptimizer()
        self.decomposition = DecompositionEngine()
        self.mdap = MDAPEngine()
        self.maker = MakerEngine()
```

##### API Integration:

OpenEvolve components will be wrapped in Hono routes:

```typescript
// /api/evolution/mutate - Triggers OpenEvolve mutation engine
// /api/evolution/decompose - Problem decomposition via OpenEvolve
// /api/evolution/optimize - MCTS/MDAP optimization
// /api/evolution/synthesize - Maker-based solution generation
// /api/evolution/adversarial - Run adversarial team workflow
// /api/evolution/gauntlet - Run gauntlet evaluation on finalists
// /api/evolution/verify - Gold team verification for final winners
```

### 2.3 Technology Stack (BubbleLab-Based)

#### 2.3.1 Frontend Stack (Extended from BubbleLab)
- **Framework**: React 19.1.0 (from BubbleLab) + TypeScript 5.8+
- **Visualization**: @xyflow/react v12.8.2 (adapted from FlowVisualizer) for evolution trees with mitosis animations
- **State Management**: Zustand v4.5.7 (from BubbleLab) with evolution-specific stores
- **Routing**: TanStack Router v1.133.28 (from BubbleLab) for evolution page navigation
- **UI Components**: Tailwind CSS v4.1.11 (from BubbleLab) + shadcn/ui
- **Build Tool**: Vite v7.0.4 (from BubbleLab)
- **Authentication**: Clerk React v5.46.0 (from BubbleLab) - reuse existing auth
- **Real-time**: Extend execution log streaming infrastructure or add Socket.io Client
- **Testing**: Vitest v3.2.4 + React Testing Library (from BubbleLab)

#### 2.3.2 Backend Stack (Extended from BubbleLab)
- **API Runtime**: Bun (from BubbleLab) + Hono v4.9.8 for high-performance API
- **Language**: TypeScript 5.8+ (from BubbleLab)
- **ORM**: Drizzle ORM v0.44.3 (from BubbleLab) with PostgreSQL/SQLite - extend schema for evolution
- **Authentication**: Clerk Backend v2.6.0 (from BubbleLab) - reuse existing auth middleware
- **AI Integration**: Extend @bubblelab/bubble-core AIAgentBubble for multi-provider visual LLM calls
  - @langchain/openai, @langchain/anthropic, @langchain/google-genai
  - DeepSeek via OpenRouter
- **Real-time**: Add Socket.io Server to existing Hono setup for evolution events
- **Job Queue**: Add Bull/Redis for evolution job management (extend existing Redis setup)

#### 2.3.3 Infrastructure (Extended from BubbleLab)
- **Containerization**: Docker + Docker Compose (from BubbleLab) - add screenshot service
- **Orchestration**: Kubernetes (production) - extend BubbleLab deployment patterns
- **Screenshots**: Add NEW Puppeteer service for HTML → Screenshot rendering
- **Database**: PostgreSQL 15+ (from BubbleLab) - extend existing schema with evolution tables
- **Cache**: Redis 7+ - extend existing Redis setup for evolution caching
- **Storage**: AWS S3/Cloudflare R2 via existing Storage service (from BubbleLab)
- **CDN**: Cloudflare (from BubbleLab deployment)
- **Hosting**: Extend existing BubbleLab deployment infrastructure

#### 2.3.4 AI/ML Services (Extended from BubbleLab)
- **Visual LLM Judges** (via extended AIAgentBubble):
  - GPT-4o/GPT-5 (OpenAI) - Primary layout judge
  - Claude 3.5/4.5 Sonnet (Anthropic) - Accessibility specialist
  - Gemini 2.5 Flash/Pro (Google) - Brand alignment specialist
  - DeepSeek Chat (OpenRouter) - Cost-optimized pre-filtering
- **Fallback Strategy**: Claude Haiku 4.5 for initial cost filtering before full evaluation
- **Cost Tracking**: Extend existing BubbleLab cost infrastructure for vision tokens

#### 2.3.5 OpenEvolve Python Integration
- **Python Runtime**: Python 3.10+ required for OpenEvolve components 🔧
- **Repository Setup**: Clone OpenEvolve as git submodule in BubbleLab project
  ```bash
  cd BubbleLab
  git submodule add https://github.com/your-org/openevolve.git services/openevolve
  cd services/openevolve
  git checkout main
  ```
- **Python Dependencies** (requirements.txt):
  ```txt
  # Core dependencies
  numpy>=1.24.0
  scipy>=1.10.0
  networkx>=3.1

  # LLM Integration
  langchain>=0.1.0
  openai>=1.0.0
  anthropic>=0.8.0

  # Async support
  aiohttp>=3.9.0

  # MCTS/MDAP algorithms
  pymoo>=0.6.0
  deap>=1.3.0
  ```
- **Environment Variables**:
  - `OPENEVOLVE_PATH=./services/openevolve`
  - `PYTHON_PATH=python3`
  - `OPENEVOLVE_API_PORT=8000`
  - `OPENEVOLVE_LOG_LEVEL=info`

### 2.4 Week 1 Setup: BubbleLab + OpenEvolve Integration

#### 2.4.1 Development Environment Prerequisites
- **Node.js 20+** (for BubbleLab) 🟢
- **Python 3.10+** (for OpenEvolve) 🔧 - **NEW REQUIREMENT**
- **Bun 1.0+** (for BubbleLab runtime) 🟢
- **Docker & Docker Compose** (for local services) 🟢

#### 2.4.2 Day 1 Setup Tasks
1. **Clone BubbleLab repository** 🟢
   ```bash
   git clone https://github.com/your-org/bubblelab.git
   cd bubblelab
   ```

2. **Add OpenEvolve as git submodule** 🔧 - **NEW**
   ```bash
   git submodule add https://github.com/your-org/openevolve.git services/openevolve
   git submodule update --init --recursive
   ```

3. **Install Node.js dependencies** 🟢
   ```bash
   bun install
   ```

4. **Install Python dependencies** 🔧 - **NEW**
   ```bash
   cd services/openevolve
   pip install -r requirements.txt
   cd ../..
   ```

5. **Configure environment variables** (both platforms)
   ```bash
   # .env.local (BubbleLab root)
   VITE_API_URL=http://localhost:3001
   DATABASE_URL=postgresql://localhost:5432/bubblelab
   REDIS_URL=redis://localhost:6379

   # OpenEvolve Configuration
   OPENEVOLVE_PATH=./services/openevolve
   PYTHON_PATH=python3
   OPENEVOLVE_API_PORT=8000
   ```

6. **Start development services**
   ```bash
   # Terminal 1: BubbleLab API
   cd apps/bubblelab-api
   bun run dev

   # Terminal 2: OpenEvolve API (NEW)
   cd services/openevolve
   python -m uvicorn api.server:app --port 8000 --reload

   # Terminal 3: BubbleStudio UI
   cd apps/bubble-studio
   bun run dev
   ```

#### 2.4.3 Integration Architecture (TypeScript → Python)

```typescript
// BubbleLab (TypeScript) calls OpenEvolve (Python) via HTTP
// apps/bubblelab-api/src/routes/evolution/mutate.ts

import { OpenEvolveClient } from '@/lib/openevolve-client';

const openEvolve = new OpenEvolveClient({
  baseURL: process.env.OPENEVOLVE_API_URL || 'http://localhost:8000',
  timeout: 30000 // 30 second timeout
});

app.post('/api/evolution/mutate', async (c) => {
  const { design, criteria } = await c.req.json();

  // Call OpenEvolve mutation engine 🔧
  const mutations = await openEvolve.mutate({
    html: design.html,
    css: design.css,
    criteria: criteria,
    populationSize: 50,
    mutationRate: 0.5
  });

  return c.json(mutations);
});
```

```python
# OpenEvolve (Python) exposes HTTP API
# services/openevolve/api/server.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="OpenEvolve API")

@app.post("/mutate")
async def mutate_design(request: MutationRequest):
    """Generate mutations using OpenEvolve engine"""
    from openevolve.evolutionary_optimization import EvolutionaryOptimizer

    engine = EvolutionaryOptimizer()
    mutations = engine.mutate(
        html=request.html,
        css=request.css,
        criteria=request.criteria,
        population_size=request.population_size
    )
    return {"mutations": mutations}
```

#### 2.4.4 Week 1 Task Breakdown

| Day | Tasks | Platform | Marker |
|-----|-------|----------|--------|
| Day 1 | Clone BubbleLab repository | BubbleLab | 🟢 |
| Day 1 | Clone OpenEvolve as submodule (NEW) | OpenEvolve | 🔧 |
| Day 1 | Install Node.js dependencies | BubbleLab | 🟢 |
| Day 1 | Install Python dependencies (NEW) | OpenEvolve | 🔧 |
| Day 1 | Configure environment variables | Both | 🟢 + 🔧 |
| Day 2-3 | Extend AIAgentBubble for visual judges | BubbleLab | 🟢 |
| Day 4-5 | Create evolution UI pages | BubbleLab | 🟢 |
| Day 6-7 | Integrate OpenEvolve mutation engine | OpenEvolve | 🔧 |

---

## 3. System Components

### 3.1 Screenshot Renderer Service

**Purpose**: Render HTML/CSS to screenshots for visual evaluation

**Architecture**:
```typescript
class ScreenshotRenderer {
  private browserPool: Browser[];
  private maxConcurrent: number = 50;
  private viewport: ViewportConfig = {
    width: 1920,
    height: 1080,
    deviceScaleFactor: 1
  };

  async render(html: string, options?: RenderOptions): Promise<Buffer> {
    // 1. Get available browser from pool
    // 2. Create page with configured viewport
    // 3. Set HTML content
    // 4. Wait for network idle
    // 5. Wait for custom selectors (fonts, images)
    // 6. Capture screenshot
    // 7. Return buffer
  }

  async renderBatch(htmls: string[]): Promise<Buffer[]> {
    // Parallel rendering with concurrency control
  }
}
```

**Key Features**:
- Browser pool management (reuse Chrome instances)
- Concurrent rendering (50+ parallel)
- Network idle detection
- Custom wait conditions
- Full-page and viewport screenshots
- PNG optimization
- Error handling with retries
- Resource blocking (block ads, trackers)

**Deployment**: Docker container with Puppeteer, Chrome, and Node.js runtime

---

### 3.2 Visual LLM Judge Service

**Purpose**: Evaluate screenshot quality using multi-agent vision models

**Architecture**:
```typescript
interface JudgeAgent {
  name: string;
  model: string;
  provider: 'openai' | 'anthropic' | 'google';
  prompt: string;
  weight: number;
  costPerImage: number;
}

class VisualLLMJudge {
  private agents: JudgeAgent[] = [
    {
      name: 'LayoutAgent',
      model: 'gpt-4o',
      provider: 'openai',
      prompt: 'Evaluate layout quality...',
      weight: 0.25,
      costPerImage: 0.01
    },
    {
      name: 'AccessibilityAgent',
      model: 'claude-3-5-sonnet',
      provider: 'anthropic',
      prompt: 'Evaluate WCAG compliance...',
      weight: 0.25,
      costPerImage: 0.0075
    },
    {
      name: 'BrandAgent',
      model: 'gemini-2.0-flash',
      provider: 'google',
      prompt: 'Evaluate brand alignment...',
      weight: 0.25,
      costPerImage: 0.005
    },
    {
      name: 'ConversionAgent',
      model: 'gpt-4o',
      provider: 'openai',
      prompt: 'Evaluate conversion potential...',
      weight: 0.25,
      costPerImage: 0.01
    }
  ];

  async evaluate(
    screenshot: Buffer,
    criteria: FitnessCriteria
  ): Promise<JudgeScore[]> {
    // Parallel evaluation across all agents
    // Cost: ~$0.0325 per design for all 4 agents
  }

  async evaluateBatch(
    screenshots: Buffer[],
    criteria: FitnessCriteria
  ): Promise<JudgeScore[][]> {
    // Batch processing with rate limiting
  }

  aggregateScores(scores: JudgeScore[]): number {
    // Weighted average based on agent weights
  }
}
```

**Key Features**:
- Multi-agent consensus voting
- Rate limiting per provider
- Automatic retries with exponential backoff
- Cost tracking and budget alerts
- Fallback to cheaper models (Haiku) for initial filtering
- Response caching (same screenshot + criteria)
- Structured JSON output parsing
- Error handling for API failures

**Prompt Engineering**:
- System prompts with role definition
- Few-shot examples in prompts
- Chain-of-thought reasoning
- Structured JSON output schema
- Criteria-specific prompt templates

---

### 3.3 Mutation Engine (OpenEvolve Integration 🔧)

**Purpose**: Generate design variations through mutation operators provided by OpenEvolve

**Architecture:**
- **Provider**: 🔧 OpenEvolve (`services/openevolve/evolutionary_optimization.py`)
- **Interface**: HTTP REST API (FastAPI server on port 8000)
- **Client**: TypeScript wrapper (`apps/bubblelab-api/src/services/evolution/openevolve-client.ts`)
- **Communication**: POST /api/evolution/mutate → Python → return mutations

**OpenEvolve Components:**
1. `decomposition_engine.py` - Problem breakdown into mutatable components
2. `evolutionary_optimization.py` - Main mutation orchestration
3. `mdap_engine.py` - Multi-objective decomposition algorithm
4. `adversarial_mdap_mcts.py` - Advanced adversarial mutation strategies

**Mutation Operators (implemented in Python):**

#### 3.3.1 Color Mutations
```python
# Implemented in: services/openevolve/evolutionary_optimization.py
class ColorMutator:
  private palettes = [
    // Industry-specific palettes
    { name: 'saas-trust', colors: ['#0066FF', '#0052CC', '#003D99'] },
    { name: 'saos-playful', colors: ['#6366F1', '#8B5CF6', '#A78BFA'] },
    { name: 'ecommerce-urgent', colors: ['#FF4444', '#CC0000', '#990000'] },
    // ... 50+ curated palettes
  ];

  mutate(css: string): string[] {
    return this.palettes.map(palette =>
      css.replace(
        /--primary:\s*#[0-9A-F]{6}/gi,
        `--primary: ${palette.colors[0]}`
      ).replace(
        /--secondary:\s*#[0-9A-F]{6}/gi,
        `--secondary: ${palette.colors[1]}`
      )
    );
  }

  generateComplementary(hex: string): string {
    // Color theory algorithms
  }

  generateAnalogous(hex: string): string[] {
    // Color theory algorithms
  }
}
```

#### 3.3.2 Typography Mutations
```typescript
class TypographyMutator {
  private scales = [
    { name: 'minimal', base: 14, ratio: 1.2 },
    { name: 'modular', base: 16, ratio: 1.333 },
    { name: 'bold', base: 18, ratio: 1.5 }
  ];

  mutate(html: string): string[] {
    // Generate font-size variations
    // Generate font-weight variations
    // Generate line-height variations
    // Generate letter-spacing variations
  }
}
```

#### 3.3.3 Layout Mutations
```typescript
class LayoutMutator {
  private gridSystems = [
    { columns: 12, gap: '1rem' },
    { columns: 16, gap: '1.5rem' },
    { columns: 24, gap: '2rem' }
  ];

  private containers = [
    { maxWidth: '1200px', center: true },
    { maxWidth: '1400px', center: true },
    { maxWidth: '100%', center: false }
  ];

  mutate(html: string): string[] {
    // Grid column variations
    // Flex direction variations
    // Container width variations
    // Spacing variations
    // Component position variations
  }
}
```

#### 3.3.4 Content Mutations
```typescript
class ContentMutator {
  mutate(html: string): string[] {
    // CTA text variations
    // Heading variations
    // Content hierarchy variations
    // Trust signal additions/removals
  }
}
```

#### 3.3.5 Component Mutations
```typescript
class ComponentMutator {
  mutate(html: string): string[] {
    // Button style variations
    // Navigation position variations
    // Hero layout variations
    // Section ordering variations
  }
}
```

**Mutation Strategy**:
- Adaptive mutation rate (decreases over generations)
- Elitism (top 10% unchanged)
- Crossover (combine mutations from multiple parents)
- Smart mutations (learn from past successes)
- Constraint-aware (respect user-defined limits)

---

### 3.4 Evolution Orchestrator

**Purpose**: Coordinate the entire evolution pipeline

**Architecture:**
- **Language**: TypeScript (runs in BubbleLab API)
- **Mutation Provider**: Calls OpenEvolve Python service via HTTP 🔧
- **Evaluation Provider**: Extended AIAgentBubble 🟢
- **Storage**: BubbleLab Storage service 🟢

```typescript
class EvolutionOrchestrator {
  private openEvolve: OpenEvolveClient;  // 🔧 Python service client
  private renderer: ScreenshotRenderer;   // 🟢 BubbleLab service
  private judge: VisualLLMJudge;          // 🟢 Extended AIAgentBubble
  private storage: StorageService;        // 🟢 BubbleLab service

  async evolve(request: EvolutionRequest): Promise<EvolutionResult> {
    const history: EvolutionGeneration[] = [];
    let population = [request.seed];

    for (let gen = 0; gen < request.generations; gen++) {
      // 1. Mutate (via OpenEvolve Python service) 🔧
      const variants = await this.openEvolve.mutateBatch({
        population,
        mutationRate: request.mutationRate,
        criteria: request.criteria
      });

      // 2. Render (via BubbleLab Screenshot service) 🟢
      const screenshots = await this.renderer.renderBatch(
        variants.map(v => v.html)
      );

      // 3. Evaluate (via BubbleLab AIAgentBubble) 🟢
      const scores = await this.judge.evaluateBatch(
        screenshots,
        request.criteria
      );

      // 4. Select (NEW TypeScript logic)
      const scored = variants.map((v, i) => ({
        ...v,
        fitness: this.aggregateScore(scores[i])
      }));

      const sorted = scored.sort((a, b) => b.fitness - a.fitness);
      const survivors = sorted.slice(0, request.populationSize);

      population = survivors;

      // 5. Emit event (via BubbleLab Socket.io) 🟢
      this.eventBus.emit('generation', {
        number: gen,
        population,
        survivors,
        pruned: sorted.slice(request.populationSize)
      });

      history.push({
        number: gen,
        population,
        survivors,
        pruned: sorted.slice(request.populationSize)
      });
    }

    return {
      winner: population[0],
      history,
      metadata: this.generateMetadata(history)
    };
  }
}
```

**Key Features**:
- **Hybrid Architecture**: TypeScript orchestrator calling Python mutations 🔧
- Real-time WebSocket streaming (via BubbleLab Socket.io) 🟢
- Checkpoint/resume support
- Progress tracking
- Cost estimation
- Early stopping (convergence detection)
- Parallel population evaluation
- Evolution history persistence

---

### 3.5 Event Bus & Real-time Communication

```typescript
class EvolutionEventBus {
  private io: Socket.Server;

  broadcastGeneration(gen: EvolutionGeneration) {
    this.io.emit('generation', {
      number: gen.number,
      timestamp: Date.now(),
      population: gen.population.map(d => ({
        id: d.id,
        thumbnail: d.screenshot,
        fitness: d.fitness,
        state: d.fitness > 0.7 ? 'passed' : 'failed'
      }))
    });
  }

  broadcastComplete(result: EvolutionResult) {
    this.io.emit('complete', {
      winner: result.winner,
      totalCost: result.metadata.cost,
      duration: result.metadata.duration
    });
  }
}
```

**Existing Reference**:
- `api/gateway/routes/evolution.py` implements `/evolution/start`, `/pause`, `/resume`, `/stop` and uses `EvolutionRoomManager` for WebSocket broadcasts.
- `api/gateway/realtime/manager.py` defines ConnectionManager/RoomManager message structure.
- `api/gateway/README.md` documents WebSocket channels and event types.

---

### 3.6 Adversarial & Team System (OpenEvolve Integration 🔧)

**Purpose**: Stress-test designs with red/blue/evaluator teams and adversarial evolution to improve robustness and quality.

**OpenEvolve Assets**:
- `red_team.py`, `blue_team.py`, `evaluator_team.py` for team-driven assessments
- `red_team_coordinator.py`, `blue_team_coordinator.py`, `evaluator_team_coordinator.py` for orchestration
- `team_manager.py`, `team_assignment_engine.py` for coordination and assignment
- `adversarial_unified.py`, `adversarial_config.py`, `adversarial_analytics.py` for adversarial workflows + metrics
- `evolution.py` integrates team phases + gauntlet handoffs in the core workflow
- UI reference: `bubblelabs_evolution_integration.py`, `bubblelabs_evolution_controls.py`
- Workflow reference: `bubblelabs_integration.py` (workflow graph + status transitions)

**Integration Notes**:
- Add team configuration (models, rounds, thresholds) to `EvolutionRequest`
- Capture findings, fixes, and evaluation metrics in `EvolutionResult`
- Feed adversarial output back into selection/fitness aggregation

---

### 3.7 Decomposition Workflow Engine (OpenEvolve Integration 🔧)

**Purpose**: Decompose problems into sub-problems to guide targeted mutations and evaluation.

**OpenEvolve Assets**:
- `decomposition_engine.py` for semantic decomposition and dependency graphs
- `decomposition_engine_adaptive_enhancement.py`, `adaptive_decomposition_integration.py` for adaptive workflows
- `decomposition_maker_integration.py`, `decomposition_mdap_integration.py` for workflow integration
- `gauntlet_decomposition_integration.py` for gauntlet-aware decomposition
- `bubblelabs_nodes/decomposition_node.py` for BubbleLabs node schema + UI config

**Integration Notes**:
- Run decomposition as an optional pre-processing phase
- Persist decomposition plan/graph and use it to drive mutation focus

---

### 3.8 Gauntlet & Gold Team Verification (OpenEvolve Integration 🔧)

**Purpose**: Gate top candidates through gauntlet evaluation and gold team verification before final selection.

**OpenEvolve Assets**:
- `gauntlet_manager.py`, `adaptive_gauntlet_system.py` for multi-stage gauntlet testing
- `gauntlet_decomposition_integration.py` for decomposition-aware gauntlet runs
- `ragbits_integration/agents/gold_team_agent.py` for final verification        
- `bubblelabs_nodes/gauntlet_node.py` for BubbleLabs node schema + UI config    

**Integration Notes**:
- Configure gauntlet presets and thresholds per evolution request
- Store gauntlet outcomes and gold team reports with final winners

---

## 4. Data Models & Schemas

### 4.1 Core Entities

#### 4.1.1 Design
```typescript
interface Design {
  id: string; // UUID
  html: string;
  css: string;
  screenshot?: string; // Base64 or URL
  thumbnail?: string; // Base64 or URL
  generation: number;
  parentId?: string; // For lineage tracking
  fitness?: number; // 0-1 aggregated score
  state: 'pending' | 'evaluating' | 'passed' | 'failed' | 'winner';
  mutations: Mutation[];
  scores?: JudgeScore[];
  createdAt: Date;
  updatedAt: Date;
}

interface Mutation {
  type: 'color' | 'typography' | 'layout' | 'content' | 'component';
  description: string;
  before?: string;
  after?: string;
}
```

#### 4.1.2 Fitness Criteria
```typescript
interface FitnessCriteria {
  brand: {
    type: string; // 'B2B SaaS', 'E-commerce', 'Blog'
    vibe: string[]; // ['professional', 'trustworthy', 'innovative']
    colors?: string[]; // Brand color constraints
    fonts?: string[]; // Font constraints
  };
  audience: {
    demographic: string; // 'CTOs at Series A startups'
    ageRange?: string;
    location?: string;
  };
  goals: string[]; // ['Book demos', 'Generate leads']
  constraints: {
    sections?: string[]; // Required sections
    elements?: string[]; // Required elements
    forbidden?: string[]; // Things to avoid
  };
  weights: {
    aesthetics: number; // 0-1
    conversion: number; // 0-1
    accessibility: number; // 0-1
    brand: number; // 0-1
  };
}
```

#### 4.1.3 Judge Score
```typescript
interface JudgeScore {
  agent: string; // 'LayoutAgent', 'AccessibilityAgent', etc.
  model: string; // 'gpt-4o', 'claude-3-5-sonnet'
  score: number; // 0-1
  reasoning: string;
  improvements: string[];
  metrics: {
    hierarchy?: number;
    contrast?: number;
    alignment?: number;
    spacing?: number;
    brandFit?: number;
    conversion?: number;
  };
  cost: number; // USD
  latency: number; // ms
}
```

#### 4.1.4 Evolution Request
```typescript
interface EvolutionRequest {
  id: string;
  userId: string;
  seed: Design;
  criteria: FitnessCriteria;
  generations: number;
  populationSize: number;
  mutationRate: number;
  elitismCount: number;
  estimatedCost: number;
  maxBudget?: number;
  createdAt: Date;
}
```

#### 4.1.5 Evolution Result
```typescript
interface EvolutionResult {
  id: string;
  requestId: string;
  winner: Design;
  history: EvolutionGeneration[];
  metadata: {
    totalDesigns: number;
    totalCost: number;
    duration: number;
    convergence: number[];
    bestFitnessPerGeneration: number[];
    avgFitnessPerGeneration: number[];
  };
  createdAt: Date;
  completedAt: Date;
}
```

### 4.2 Database Schema

```sql
-- Users
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  tier VARCHAR(50) DEFAULT 'free', -- free, pro, agency, enterprise
  credits_remaining INTEGER DEFAULT 10,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Evolution Requests
CREATE TABLE evolution_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  criteria JSONB NOT NULL,
  generations INTEGER NOT NULL,
  population_size INTEGER NOT NULL,
  estimated_cost DECIMAL(10, 4),
  max_budget DECIMAL(10, 4),
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Designs
CREATE TABLE designs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id UUID REFERENCES evolution_requests(id),
  parent_id UUID REFERENCES designs(id),
  generation INTEGER NOT NULL,
  html TEXT NOT NULL,
  css TEXT NOT NULL,
  screenshot_url TEXT,
  thumbnail_url TEXT,
  fitness DECIMAL(5, 4),
  state VARCHAR(50) DEFAULT 'pending',
  mutations JSONB,
  scores JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Evolution Results
CREATE TABLE evolution_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id UUID REFERENCES evolution_requests(id),
  winner_id UUID REFERENCES designs(id),
  history JSONB NOT NULL,
  metadata JSONB NOT NULL,
  total_cost DECIMAL(10, 4),
  duration INTEGER, -- milliseconds
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP
);

-- Screenshots (separate table for caching)
CREATE TABLE screenshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  design_id UUID REFERENCES designs(id),
  html_hash VARCHAR(64) UNIQUE, -- SHA-256 for deduplication
  image_url TEXT NOT NULL,
  size_bytes INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_designs_request_generation ON designs(request_id, generation);
CREATE INDEX idx_designs_fitness ON designs(fitness DESC);
CREATE INDEX idx_screenshots_hash ON screenshots(html_hash);
```

**BubbleLab Evolution Graph Persistence (Current)**:
```sql
-- Evolution runs (per user)
CREATE TABLE evolution_runs (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  evolution_id TEXT NOT NULL,
  status TEXT NOT NULL,
  name TEXT,
  config JSONB,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

-- Evolution nodes (lineage graph)
CREATE TABLE evolution_nodes (
  id SERIAL PRIMARY KEY,
  run_id INTEGER REFERENCES evolution_runs(id),
  node_id TEXT NOT NULL,
  parent_node_id TEXT,
  generation INTEGER NOT NULL,
  status TEXT NOT NULL,
  fitness DOUBLE PRECISION,
  score DOUBLE PRECISION,
  label TEXT,
  html_asset_id INTEGER,
  thumbnail_asset_id INTEGER,
  metadata JSONB,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

-- Evolution assets (local storage for HTML + thumbnails)
CREATE TABLE evolution_assets (
  id SERIAL PRIMARY KEY,
  run_id INTEGER REFERENCES evolution_runs(id),
  user_id TEXT NOT NULL,
  kind TEXT NOT NULL, -- html | thumbnail
  content_type TEXT NOT NULL,
  file_path TEXT NOT NULL,
  size INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL
);
```

---

## 5. API Design

### 5.1 REST API Endpoints

#### Evolution Management
```typescript
POST   /api/evolution                 // Start new evolution
GET    /api/evolution/:id             // Get evolution status
DELETE /api/evolution/:id             // Cancel evolution
GET    /api/evolution/:id/history     // Get full evolution history
GET    /api/evolution/:id/export      // Export result (HTML/PDF)
```

**Existing Reference**:
- `api/gateway/routes/evolution.py` exposes `/evolution/start`, `/pause`, `/resume`, `/stop`, and list/status endpoints in FastAPI.

#### Adversarial & Evaluation
```typescript
POST   /api/evolution/:id/adversarial // Run adversarial team workflow
POST   /api/evolution/:id/gauntlet    // Run gauntlet evaluation
POST   /api/evolution/:id/verify      // Gold team verification
POST   /api/evolution/decompose       // Generate decomposition plan
```

#### Design Management
```typescript
GET    /api/designs/:id               // Get design details
GET    /api/designs/:id/screenshot    // Get design screenshot
GET    /api/designs/:id/compare/:id2  // Compare two designs
POST   /api/designs/:id/fork          // Fork design for new evolution
```

#### User Management
```typescript
GET    /api/user/profile              // Get user profile
GET    /api/user/credits              // Get credit balance
GET    /api/user/evolutions           // Get user's evolutions
PATCH  /api/user/tier                 // Upgrade tier
```

#### Templates & Presets
```typescript
GET    /api/templates                 // List templates
GET    /api/templates/:id             // Get template details
POST   /api/templates                 // Create custom template
```

### 5.2 WebSocket Events

#### Client → Server
```typescript
socket.emit('start_evolution', {
  seed: Design,
  criteria: FitnessCriteria,
  generations: number,
  populationSize: number
});

socket.emit('cancel_evolution', { id: string });

socket.emit('subscribe_evolution', { id: string });
```

#### Server → Client
```typescript
socket.on('generation_start', {
  number: number,
  populationSize: number
});

socket.on('design_evaluated', {
  id: string,
  fitness: number,
  state: 'passed' | 'failed',
  scores: JudgeScore[]
});

socket.on('generation_complete', {
  number: number,
  survivors: Design[],
  pruned: Design[]
});

socket.on('evolution_complete', {
  winner: Design,
  history: EvolutionGeneration[],
  metadata: EvolutionMetadata
});

socket.on('error', {
  message: string,
  code: string
});
```

### 5.3 GraphQL Schema (Optional)

```graphql
type Evolution {
  id: ID!
  user: User!
  criteria: FitnessCriteria!
  status: EvolutionStatus!
  generations: Int!
  populationSize: Int!
  result: EvolutionResult
  createdAt: DateTime!
  completedAt: DateTime
}

type Design {
  id: ID!
  evolution: Evolution!
  generation: Int!
  parent: Design
  children: [Design!]!
  html: String!
  css: String!
  screenshot: String
  thumbnail: String
  fitness: Float
  state: DesignState!
  mutations: [Mutation!]!
  scores: [JudgeScore!]!
}

type Query {
  evolution(id: ID!): Evolution
  evolutions(
    userId: ID
    status: EvolutionStatus
    limit: Int
    offset: Int
  ): [Evolution!]!
  design(id: ID!): Design
  compareDesigns(id1: ID!, id2: ID!): DesignComparison
}

type Mutation {
  startEvolution(input: EvolutionInput!): Evolution!
  cancelEvolution(id: ID!): Boolean!
  forkDesign(id: ID!): Evolution!
  exportEvolution(id: ID!, format: ExportFormat!): String
}
```

---

## 6. Visual LLM Integration

### 6.1 Judge Agent Configuration

#### Layout Agent
```typescript
const LayoutAgent: JudgeAgent = {
  name: 'LayoutAgent',
  model: 'gpt-4o',
  prompt: `You are an expert UI/UX designer evaluating web design layouts.

Evaluate the provided screenshot on layout quality (0-1 scale):

Criteria:
1. Visual Hierarchy (0-1): Is the most important content most prominent?
2. Whitespace Usage (0-1): Is whitespace used effectively to reduce cognitive load?
3. Grid Alignment (0-1): Are elements aligned to a consistent grid?
4. Balance (0-1): Is the visual weight balanced across the design?
5. Scanability (0-1): Can users quickly scan and understand the content?

Return JSON:
{
  "scores": {
    "hierarchy": 0.X,
    "whitespace": 0.X,
    "alignment": 0.X,
    "balance": 0.X,
    "scanability": 0.X
  },
  "overall": 0.X,
  "reasoning": "3-4 sentence explanation",
  "improvements": ["specific improvement 1", "specific improvement 2"]
}`,
  weight: 0.25
};
```

#### Accessibility Agent
```typescript
const AccessibilityAgent: JudgeAgent = {
  name: 'AccessibilityAgent',
  model: 'claude-3-5-sonnet',
  prompt: `You are a WCAG accessibility expert.

Evaluate the design for accessibility compliance (0-1 scale):

Criteria:
1. Color Contrast (0-1): Does text meet WCAG AA standards (4.5:1 for normal text)?
2. Touch Targets (0-1): Are interactive elements at least 44x44px?
3. Font Size (0-1): Is body text at least 16px (12pt)?
4. Focus Indicators (0-1): Are interactive elements clearly identifiable?
5. Semantic Structure (0-1): Is there clear heading hierarchy?

Return JSON:
{
  "scores": {
    "contrast": 0.X,
    "touchTargets": 0.X,
    "fontSize": 0.X,
    "focusIndicators": 0.X,
    "semanticStructure": 0.X
  },
  "overall": 0.X,
  "reasoning": "...",
  "improvements": ["...", "..."]
}`,
  weight: 0.25
};
```

#### Brand Alignment Agent
```typescript
const BrandAgent: JudgeAgent = {
  name: 'BrandAgent',
  model: 'gemini-2.0-flash',
  prompt: `You are a brand consistency expert.

Evaluate brand alignment (0-1 scale):

Brand Context: {brand}
Target Audience: {audience}
Brand Attributes: {attributes}

Criteria:
1. Color Harmony (0-1): Do colors align with brand palette?
2. Typography Fit (0-1): Do fonts match brand personality?
3. Tone Consistency (0-1): Does the design feel like {brand}?
4. Audience Appeal (0-1): Would {audience} find this credible?
5. Professionalism (0-1): Does it meet industry standards?

Return JSON:
{
  "scores": {
    "colorHarmony": 0.X,
    "typographyFit": 0.X,
    "toneConsistency": 0.X,
    "audienceAppeal": 0.X,
    "professionalism": 0.X
  },
  "overall": 0.X,
  "reasoning": "...",
  "improvements": ["...", "..."]
}`,
  weight: 0.25
};
```

#### Conversion Agent
```typescript
const ConversionAgent: JudgeAgent = {
  name: 'ConversionAgent',
  model: 'gpt-4o',
  prompt: `You are a conversion rate optimization (CRO) expert.

Evaluate conversion potential (0-1 scale):

Goals: {goals}

Criteria:
1. CTA Prominence (0-1): Is the primary CTA immediately visible?
2. Value Proposition (0-1): Is the value proposition clear above the fold?
3. Trust Signals (0-1): Are trust badges, testimonials, or social proof present?
4. Friction Reduction (0-1): Is the path to conversion simple and obvious?
5. Urgency/Incentive (0-1): Is there a reason to act now?

Return JSON:
{
  "scores": {
    "ctaProminence": 0.X,
    "valueProp": 0.X,
    "trustSignals": 0.X,
    "frictionReduction": 0.X,
    "urgency": 0.X
  },
  "overall": 0.X,
  "reasoning": "...",
  "improvements": ["...", "..."]
}`,
  weight: 0.25
};
```

### 6.2 Aggregation Strategy

```typescript
function aggregateScores(
  agentScores: JudgeScore[],
  userWeights: FitnessCriteria['weights']
): number {
  // User-defined weights override default agent weights
  const weights = {
    aesthetics: userWeights.aesthetics, // Affects LayoutAgent
    conversion: userWeights.conversion, // Affects ConversionAgent
    accessibility: userWeights.accessibility, // Affects AccessibilityAgent
    brand: userWeights.brand // Affects BrandAgent
  };

  // Weighted average
  return (
    agentScores[0].overall * weights.aesthetics +
    agentScores[1].overall * weights.accessibility +
    agentScores[2].overall * weights.brand +
    agentScores[3].overall * weights.conversion
  );
}
```

### 6.3 Cost Optimization Strategy

```typescript
class CostOptimizedJudge {
  async evaluate(screenshot: Buffer, criteria: FitnessCriteria) {
    // Round 1: Cheap filter (Claude Haiku - $0.001/image)
    const quickScore = await this.quickEvaluate(screenshot, criteria);

    // If quick score < 0.4, reject without expensive evaluation
    if (quickScore.overall < 0.4) {
      return quickScore;
    }

    // Round 2: Full evaluation (4 agents - $0.0325/image)
    const fullScore = await this.fullEvaluate(screenshot, criteria);

    return fullScore;
  }
}
```

---

## 7. Mutation Engine Design (OpenEvolve Integration 🔧)

**Overview**: The mutation engine is provided by OpenEvolve and exposes advanced genetic operators through a Python HTTP service.

### 7.1 Mutation Strategy Matrix (from OpenEvolve)

| Generation | Mutation Rate | Elitism | Crossover | New Blood |
|------------|---------------|---------|-----------|-----------|
| 1-3        | 0.8           | 10%     | 20%       | 30%       |
| 4-7        | 0.6           | 15%     | 30%       | 20%       |
| 8-10       | 0.4           | 20%     | 40%       | 10%       |
| 11+        | 0.2           | 25%     | 50%       | 5%        |

### 7.2 Adaptive Mutations

```typescript
class AdaptiveMutationEngine {
  private mutationHistory: Map<string, number> = new Map();

  getSuccessfulMutations(generation: number): MutationType[] {
    // Track which mutations led to survivors
    // Increase probability of successful mutations
    // Decrease probability of failed mutations
  }

  detectConvergence(population: Design[]): boolean {
    // If fitness variance < 0.05, population has converged
    const fitnesses = population.map(d => d.fitness);
    const variance = Math.variance(fitnesses);
    return variance < 0.05;
  }

  introduceNovelty(population: Design[]): Design[] {
    // If converged, introduce radical mutations
    // Completely new color palettes
    // Radical layout changes
    // New component structures
  }
}
```

### 7.3 Constraint-Aware Mutations

```typescript
class ConstraintAwareMutator {
  mutate(design: Design, criteria: FitnessCriteria): Design[] {
    const mutations: Design[] = [];

    // Respect brand color constraints
    if (criteria.brand.colors) {
      mutations.push(
        ...this.colorMutator.mutateWithConstraints(
          design,
          criteria.brand.colors
        )
      );
    }

    // Respect required sections
    if (criteria.constraints.sections) {
      mutations.push(
        ...this.layoutMutator.mutatePreservingSections(
          design,
          criteria.constraints.sections
        )
      );
    }

    // Never add forbidden elements
    if (criteria.constraints.forbidden) {
      mutations = mutations.filter(m =>
        !this.containsForbidden(m, criteria.constraints.forbidden)
      );
    }

    return mutations;
  }
}
```

---

## 8. Evolution Orchestrator

### 8.1 Pipeline Architecture

```typescript
class EvolutionPipeline {
  private stages: PipelineStage[] = [
    new ValidationStage(),
    new MutationStage(),
    new RenderingStage(),
    new EvaluationStage(),
    new SelectionStage(),
    new EmissionStage()
  ];

  async process(request: EvolutionRequest): Promise<EvolutionResult> {
    let context: PipelineContext = {
      request,
      population: [request.seed],
      generation: 0
    };

    for (let i = 0; i < request.generations; i++) {
      for (const stage of this.stages) {
        context = await stage.execute(context);
      }
      context.generation++;
    }

    return context.result;
  }
}
```

### 8.2 Checkpoint & Resume

```typescript
class CheckpointManager {
  async saveCheckpoint(context: PipelineContext) {
    await redis.setex(
      `checkpoint:${context.request.id}`,
      86400, // 24 hour TTL
      JSON.stringify({
        population: context.population,
        generation: context.generation,
        metadata: context.metadata
      })
    );
  }

  async loadCheckpoint(requestId: string): Promise<PipelineContext> {
    const data = await redis.get(`checkpoint:${requestId}`);
    return JSON.parse(data);
  }
}
```

### 8.3 Progress Tracking

```typescript
class ProgressTracker {
  private progress: Map<string, EvolutionProgress> = new Map();

  update(requestId: string, stage: string, progress: number) {
    this.progress.set(requestId, {
      stage,
      progress,
      timestamp: Date.now()
    });

    // Broadcast to client
    this.io.emit('progress', { requestId, stage, progress });
  }

  getETA(requestId: string): number {
    // Estimate time remaining based on progress
  }
}
```

### 8.4 Adversarial & Team Phases

```typescript
// Optional adversarial phase between generations
if (request.adversarial?.enabled) {
  const teamResult = await adversarialEngine.run({
    content: currentBest.html,
    rounds: request.adversarial.rounds,
    models: request.adversarial.models
  });

  context.metrics.adversarial = teamResult.metrics;
  context.findings = teamResult.findings;
}
```

**Notes**:
- Uses `adversarial_unified.py` with Red/Blue/Evaluator team modules
- Results feed back into fitness aggregation and selection

---

### 8.5 Gauntlet & Gold Team Gates

```typescript
// Final gating before winner selection
if (request.gauntlet?.enabled) {
  const gauntletResult = await gauntlet.run(context.candidates);
  context.metrics.gauntlet = gauntletResult.metrics;
}

if (request.goldTeam?.enabled) {
  const verification = await goldTeam.verify(context.winner);
  context.metrics.goldTeam = verification.summary;
}
```

**Notes**:
- Uses `gauntlet_manager.py` and `gold_team_agent.py`
- Stores gate outcomes in `EvolutionResult`

---

### 8.6 Decomposition Preprocessing

```typescript
if (request.decomposition?.enabled) {
  const plan = await decompositionEngine.decompose(request.problem);
  context.decompositionPlan = plan;
}
```

**Notes**:
- Uses `decomposition_engine.py` to generate sub-problem plans
- Plan influences mutation focus and evaluation criteria

---

## 9. Frontend & Visualization

### 9.1 Mitosis Animation Architecture

**Existing Reference**:
- `bubblelabs-mitosis-plugin/src/openevolve-evolution-integration.ts` provides event-to-animation wiring for evolution, decomposition, and adversarial flows.

```typescript
// D3 Force-Directed Graph Setup
const simulation = d3.forceSimulation<DesignBubble>(nodes)
  .force('charge', d3.forceManyBody().strength(-30))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(d => d.r + 2))
  .force('link', d3.forceLink(links).id(d => d.id).distance(100));

// Bubble State Transitions
function animateBubbleSplit(parent: DesignBubble) {
  // Phase 1: Pulse
  parent.transition()
    .duration(500)
    .attr('r', parent.r * 1.5)
    .style('stroke-width', 4);

  // Phase 2: Spawn children
  const children = spawnChildren(parent);
  children.forEach(child => {
    child
      .attr('opacity', 0)
      .attr('r', 0)
      .transition()
      .duration(1000)
      .attr('opacity', 1)
      .attr('r', child.targetR);
  });

  // Phase 3: Parent fade
  parent.transition()
    .delay(500)
    .duration(500)
    .attr('opacity', 0.3);
}

// Evaluation Animation
function animateEvaluation(bubble: DesignBubble, result: JudgeScore) {
  const color = result.overall > 0.7 ? '#22c55e' : '#ef4444';

  bubble.transition()
    .duration(1000)
    .style('fill', color)
    .style('stroke', color)
    .style('stroke-width', 4);

  if (result.overall <= 0.7) {
    bubble.transition()
      .delay(2000)
      .duration(1000)
      .attr('opacity', 0)
      .attr('r', 0);
  }
}
```

### 9.2 Interactive Features

```typescript
// Hover: Full-size preview
bubble.on('mouseenter', (event, d) => {
  tooltip
    .html(`
      <img src="${d.screenshot}" />
      <div>Fitness: ${(d.fitness * 100).toFixed(1)}%</div>
      <div>${d.mutations.map(m => m.description).join('<br>')}</div>
    `)
    .style('opacity', 1);
});

// Click: Detailed view
bubble.on('click', (event, d) => {
  openModal({
    design: d,
    scores: d.scores,
    reasoning: d.scores.map(s => s.reasoning).join('\n\n')
  });
});

// Double-click: Export design
bubble.on('dblclick', (event, d) => {
  downloadHTML(d);
  downloadCSS(d);
});
```

**Current BubbleLab Implementation (v0)**:
- Split-view layout with collapsible settings panel + right-side inspector.
- ReactFlow evolution graph with selection, minimap, and smoothstep edges.
- Live HTML render in inspector/modal (iframe) with cached thumbnail fallback.
- Preview mode toggle (live vs cached) shared between inspector and modal.
- Cached preview storage toggle (disable to reduce local disk usage).
- Storage usage indicator + purge cached previews control.
- Modal open on double-click and keyboard shortcuts (Enter/Space open, Esc close).
- Hover tooltip preview + generation timeline scrubber with play/pause.
- Winner/results panel with fitness progression + lineage highlights.
- Comparison view for side-by-side variant review and export buttons.
- Evolution run history list with load/delete controls.
- Pause/resume controls and WebSocket reconnect prompt for live runs.

### 9.3 Timeline Scrubber

```typescript
class TimelineScrubber {
  private generations: EvolutionGeneration[] = [];

  scrubToGeneration(gen: number) {
    // Update visualizations to show generation state
    // Reconstruct force graph from history
    // Animate transition from current to target generation
  }

  playEvolution() {
    // Auto-play through all generations
    // 2 seconds per generation
  }

  exportVideo() {
    // Capture evolution as MP4
  }
}
```

### 9.4 UI Components

#### Control Panel
```typescript
<ControlPanel>
  <SeedInput
    type="file"
    accept=".html,.png"
    onUpload={handleSeedUpload}
  />

  <CriteriaBuilder
    brand={criteria.brand}
    audience={criteria.audience}
    goals={criteria.goals}
    onChange={setCriteria}
  />

  <EvolutionSettings
    generations={[5, 10, 15, 20]}
    populationSize={[25, 50, 100]}
    budget={maxBudget}
    onChange={setSettings}
  />

  <StartButton
    estimatedCost={estimatedCost}
    onClick={startEvolution}
  />
</ControlPanel>
```

**Existing Reference**:
- `OpenEvolve-Plugin/src/schemas/evolution.ts`, `adversarial.ts`, `decomposition.ts` provide parameter schema definitions.
- `OpenEvolve-Plugin/src/stores/evolutionStore.ts` provides Zustand state patterns for evolution + adversarial UI.
- `BubbleLab/apps/bubble-studio/src/lib/evolution/schemas.ts` + `BubbleLab/apps/bubble-studio/src/stores/evolutionSettingsStore.ts` mirror schemas/store for the BubbleLab UI.
- `BubbleLab/apps/bubble-studio/src/services/evolutionApi.ts` + `BubbleLab/apps/bubble-studio/src/hooks/useEvolutionWebSocket.ts` wire `/evolution/start` and WebSocket progress into the settings UI.
- `BubbleLab/apps/bubble-studio/src/pages/EvolutionSettingsPage.tsx`, `BubbleLab/apps/bubble-studio/src/pages/EvolutionInsightsPage.tsx`, `/routes/evolution*.tsx` provide the wired settings/insights shells.
- `bubblelabs_integration.py` defines workflow graph structures and UI data models (BubbleLab UI reference).

**BubbleLab Wiring Files (Current)**:
- `BubbleLab/apps/bubble-studio/package.json` (cross-platform dev/build scripts)
- `BubbleLab/apps/bubble-studio/scripts/copy-bubble-artifacts.cjs`
- `BubbleLab/apps/bubble-studio/src/types/evolution.ts`
- `BubbleLab/apps/bubble-studio/src/lib/evolution/schemas.ts`
- `BubbleLab/apps/bubble-studio/src/stores/evolutionSettingsStore.ts`
- `BubbleLab/apps/bubble-studio/src/stores/evolutionRuntimeStore.ts`
- `BubbleLab/apps/bubble-studio/src/stores/evolutionGraphStore.ts`
- `BubbleLab/apps/bubble-studio/src/components/evolution/EvolutionParameterForm.tsx`
- `BubbleLab/apps/bubble-studio/src/components/evolution/EvolutionGraphView.tsx`
- `BubbleLab/apps/bubble-studio/src/components/evolution/EvolutionInspectorPanel.tsx`
- `BubbleLab/apps/bubble-studio/src/components/evolution/EvolutionNode.tsx`
- `BubbleLab/apps/bubble-studio/src/components/evolution/EvolutionPreviewModal.tsx`
- `BubbleLab/apps/bubble-studio/src/pages/EvolutionSettingsPage.tsx`
- `BubbleLab/apps/bubble-studio/src/pages/EvolutionInsightsPage.tsx`
- `BubbleLab/apps/bubble-studio/src/services/evolutionApi.ts`
- `BubbleLab/apps/bubble-studio/src/services/evolutionGraphApi.ts`
- `BubbleLab/apps/bubble-studio/src/hooks/useEvolutionWebSocket.ts`
- `BubbleLab/apps/bubble-studio/src/utils/evolutionPreview.ts`
- `BubbleLab/apps/bubble-studio/src/routes/evolution.tsx`
- `BubbleLab/apps/bubble-studio/src/routes/evolution.insights.tsx`
- `BubbleLab/apps/bubble-studio/src/components/Sidebar.tsx`
- `BubbleLab/apps/bubble-studio/src/routeTree.gen.ts` (generated)

#### Adversarial & Team Dashboard
```typescript
<EvolutionInsights>
  <InsightsTabs tabs={['Evolution', 'Adversarial', 'Gauntlet', 'History']} />
  <KPIHeader metrics={kpiMetrics} />
  <TeamFindingsPanel findings={teamFindings} />
  <GauntletResultsTable results={gauntletResults} />
  <GoldTeamReport report={goldTeamReport} />
  <DecompositionPlanView plan={decompositionPlan} />
</EvolutionInsights>
```

#### Fitness Panel
```typescript
<FitnessPanel>
  <AgentScores
    agents={scores}
    highlighted="LayoutAgent"
  />

  <ReasoningDisplay
    text={selectedDesign.scores[0].reasoning}
  />

  <ImprovementsList
    items={selectedDesign.scores[0].improvements}
  />

  <ComparisonView
    before={parentDesign}
    after={selectedDesign}
    diff={calculateDiff(parentDesign, selectedDesign)}
  />
</FitnessPanel>
```

---

## 10. Infrastructure & DevOps

### 10.1 Container Architecture (Hybrid: TypeScript + Python)

**Services:**
- 🟢 **bubble-studio**: React 19 + Vite frontend (TypeScript)
- 🟢 **bubblelab-api**: Hono + Bun API (TypeScript)
- 🔧 **openevolve-api**: FastAPI Python service (NEW - Mutation engine)
- 🟢 **screenshot-renderer**: Puppeteer service (Node.js)
- 🟢 **postgres**: PostgreSQL 15 database
- 🟢 **redis**: Redis 7 cache/queue
- 🟢 **storage**: S3/Cloudflare R2

```yaml
# docker-compose.yml (BubbleLab + OpenEvolve)
version: '3.8'

services:
  # 🟢 Frontend (BubbleLab)
  bubble-studio:
    build: ./apps/bubble-studio
    ports: ["3000:3000"]
    environment:
      - VITE_API_URL=http://bubblelab-api:3001
      - VITE_WS_URL=ws://bubblelab-api:3001

  # 🟢 API Gateway (BubbleLab - Extended)
  bubblelab-api:
    build: ./apps/bubblelab-api
    ports: ["3001:3001"]
    environment:
      - DATABASE_URL=postgresql://postgres:5432/bubblelab
      - REDIS_URL=redis://redis:6379
      - OPENEVOLVE_API_URL=http://openevolve:8000
      - SCREENSHOT_SERVICE_URL=http://screenshot-renderer:8002
    depends_on: [postgres, redis, openevolve, screenshot-renderer]

  # 🔧 OpenEvolve Mutation Engine (Python)
  openevolve:
    build: ./services/openevolve
    ports: ["8000:8000"]
    environment:
      - PYTHON_PATH=/usr/local/bin/python3
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - MUTATION_RATE=0.5
      - POPULATION_SIZE=50
    volumes:
      - ./services/openevolve:/app
    command: python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

  # 🟢 Screenshot Renderer (NEW)
  screenshot-renderer:
    build: ./services/screenshot-renderer
    ports: ["8002:8002"]
    deploy:
      replicas: 5
    environment:
      - CONCURRENT_LIMIT=10
      - CHROME_PATH=/usr/bin/chromium

  # 🟢 Database (BubbleLab - Extended)
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=bubblelab
      - POSTGRES_USER=bubblelab
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # 🟢 Cache (BubbleLab)
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

**Service Boundaries:**
- 🟢 **TypeScript Stack**: bubble-studio, bubblelab-api, screenshot-renderer
- 🔧 **Python Stack**: openevolve (mutation engine)
- 🔄 **HTTP Communication**: bubblelab-api → openevolve (port 8000)

### 10.2 Kubernetes Configuration (Production)

**Deployment: OpenEvolve Python Service** (NEW)

```yaml
# screenshot-renderer-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: screenshot-renderer
spec:
  replicas: 10 # Horizontal scaling
  selector:
    matchLabels:
      app: screenshot-renderer
  template:
    metadata:
      labels:
        app: screenshot-renderer
    spec:
      containers:
      - name: renderer
        image: evolution/screenshot-renderer:latest
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: CONCURRENT_LIMIT
          value: "10"
        - name: CHROME_PATH
          value: "/usr/bin/chromium"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: screenshot-renderer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: screenshot-renderer
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 10.3 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: npm ci
      - run: npm test
      - run: npm run lint

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: evolution/api:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: aws-actions/kubectl@v1
      - run: kubectl set image deployment/api-api-gateway api=evolution/api:${{ github.sha }}
```

---

## 11. Security & Privacy

### 11.1 Authentication & Authorization

- BubbleLab UI uses Clerk; API Gateway accepts Clerk JWTs (RS256) via JWKS.
- Internal service tokens remain HS256 (for tests, service-to-service calls).
- Normalize identity from token `sub` claim for consistent user tracking.

```env
# Clerk verification (used for BubbleLab UI tokens)
CLERK_ISSUER=https://your-instance.clerk.accounts.dev
CLERK_JWKS_URL=
CLERK_AUDIENCE=

# Internal service tokens (HS256)
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
```

```typescript
// Role-based access control
enum UserRole {
  FREE = 'free',
  PRO = 'pro',
  AGENCY = 'agency',
  ENTERPRISE = 'enterprise'
}

interface RateLimits {
  [UserRole.FREE]: { evolutions: 5, designsPerEvolution: 50 };
  [UserRole.PRO]: { evolutions: -1, designsPerEvolution: 200 }; // unlimited
  [UserRole.AGENCY]: { evolutions: -1, designsPerEvolution: 500 };
  [UserRole.ENTERPRISE]: { evolutions: -1, designsPerEvolution: 1000 };
}
```

### 11.2 Data Privacy

- Screenshots stored in encrypted S3 buckets
- User HTML/CSS encrypted at rest
- Automatic data deletion after 90 days (configurable)
- GDPR compliance (right to deletion)
- No training on user designs without explicit consent

### 11.3 API Security

```typescript
// Rate limiting
import rateLimit from 'express-rate-limit';

const evolutionLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10, // 10 evolution starts per minute
  standardHeaders: true,
  legacyHeaders: false,
});

// Input validation
import { z } from 'zod';

const EvolutionRequestSchema = z.object({
  seed: z.object({
    html: z.string().max(100_000),
    css: z.string().max(50_000)
  }),
  criteria: FitnessCriteriaSchema,
  generations: z.number().min(1).max(50),
  populationSize: z.number().min(10).max(1000)
});

// SQL injection prevention (use parameterized queries)
// XSS prevention (sanitize HTML/CSS inputs)
// CSRF protection (state tokens)
```

---

## 12. Performance Optimization

### 12.1 Screenshot Caching

```typescript
class ScreenshotCache {
  async get(html: string): Promise<Buffer | null> {
    const hash = this.hashHTML(html);
    return await redis.getBuffer(`screenshot:${hash}`);
  }

  async set(html: string, screenshot: Buffer) {
    const hash = this.hashHTML(html);
    await redis.setex(`screenshot:${hash}`, 86400, screenshot); // 24h TTL
  }

  private hashHTML(html: string): string {
    return crypto.createHash('sha256').update(html).digest('hex');
  }
}
```

### 12.2 Parallel Processing

```typescript
class ParallelProcessor {
  async process<T, R>(
    items: T[],
    processor: (item: T) => Promise<R>,
    concurrency: number
  ): Promise<R[]> {
    const chunks = _.chunk(items, concurrency);
    const results: R[] = [];

    for (const chunk of chunks) {
      const chunkResults = await Promise.all(
        chunk.map(item => processor(item))
      );
      results.push(...chunkResults);
    }

    return results;
  }
}
```

### 12.3 Database Optimization

```sql
-- Indexes for common queries
CREATE INDEX idx_designs_generation_fitness
ON designs(request_id, generation, fitness DESC);

CREATE INDEX idx_evolution_requests_user_created
ON evolution_requests(user_id, created_at DESC);

-- Partitioning for large tables
CREATE TABLE designs_partitioned (
  -- schema
) PARTITION BY RANGE (created_at);

CREATE TABLE designs_gen_1 PARTITION OF designs_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

---

## 13. Testing Strategy

### 13.1 Unit Testing

```typescript
// Mutation engine tests
describe('ColorMutator', () => {
  it('should generate 10 color variations', () => {
    const mutator = new ColorMutator();
    const mutations = mutator.mutate(baseCSS);
    expect(mutations).toHaveLength(10);
  });

  it('should preserve CSS structure', () => {
    const mutator = new ColorMutator();
    const mutations = mutator.mutate(baseCSS);
    mutations.forEach(m => {
      expect(m).toMatch(/background:\s*#[0-9A-F]{6}/);
    });
  });
});

// LLM judge tests
describe('VisualLLMJudge', () => {
  it('should return score between 0 and 1', async () => {
    const judge = new VisualLLMJudge();
    const score = await judge.evaluate(mockScreenshot, mockCriteria);
    expect(score.overall).toBeGreaterThanOrEqual(0);
    expect(score.overall).toBeLessThanOrEqual(1);
  });
});
```

### 13.2 Integration Testing

```typescript
describe('Evolution Pipeline', () => {
  it('should evolve designs for 3 generations', async () => {
    const orchestrator = new EvolutionOrchestrator();
    const request = mockEvolutionRequest({
      generations: 3,
      populationSize: 10
    });

    const result = await orchestrator.evolve(request);

    expect(result.history).toHaveLength(3);
    expect(result.winner.fitness).toBeGreaterThan(0);
  });
});
```

### 13.3 E2E Testing

```typescript
describe('Evolution UI', () => {
  it('should complete full evolution flow', async () => {
    await page.goto('/evolution');
    await page.fill('[name="brand"]', 'B2B SaaS');
    await page.fill('[name="audience"]', 'CTOs at startups');
    await page.click('button:has-text("Start Evolution")');

    // Wait for completion
    await page.waitForSelector('[data-testid="evolution-complete"]');

    // Verify winner displayed
    const winner = await page.textContent('[data-testid="winner-design"]');
    expect(winner).toBeTruthy();
  });
});
```

---

## 14. Deployment Strategy

### 14.1 Phased Rollout

**Phase 1: Alpha (Week 1-2)**
- 10 friendly users
- Single instance deployment
- Feature flag: evolution (off by default)
- Manual monitoring

**Phase 2: Beta (Week 3-4)**
- 100 users (waitlist)
- 3-instance deployment
- Cost tracking & budget limits
- Automated monitoring

**Phase 3: Public Launch (Week 5)**
- Product Hunt launch
- Auto-scaling enabled
- Full monitoring stack
- 24/7 on-call rotation

**Phase 4: Scale (Week 6+)**
- Horizontal pod autoscaling
- CDN integration
- Multi-region deployment
- Enterprise features

### 14.2 Feature Flags

```typescript
// Posthog/LaunchDarkly integration
const flags = {
  evolution: true,
  parallelRendering: true,
  costOptimization: false, // Beta feature
  realTrafficValidation: false, // Enterprise only
  customJudges: false // Enterprise only
};

if (flags.costOptimization) {
  // Use cheap Haiku filtering
} else {
  // Full evaluation for all
}
```

---

## 15. Monitoring & Observability

### 15.1 Metrics

**System Metrics**:
- CPU, memory, disk usage per service
- Request latency (p50, p95, p99)
- Error rate by endpoint
- Queue depth (evolution jobs)
- Cache hit rate

**Business Metrics**:
- Evolutions started/completed
- Average cost per evolution
- User retention (day 1, 7, 30)
- Conversion rate (free → paid)
- Feature usage

**Quality Metrics**:
- LLM judge response time
- Screenshot render time
- Evolution completion time
- Failed evolution rate
- User satisfaction (NPS)

### 15.2 Logging

```typescript
import { Logger } from 'pino';

const logger = Logger({
  name: 'evolution-api',
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
  },
  serializers: {
    req: pino.stdSerializers.req,
    res: pino.stdSerializers.res,
    err: pino.stdSerializers.err,
  },
});

logger.info({
  requestId: req.id,
  userId: req.user.id,
  action: 'evolution_started',
  generations: req.body.generations,
  estimatedCost: calculateCost(req.body)
});
```

### 15.3 Alerting

```yaml
# Prometheus alerts
groups:
- name: evolution_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(errors_total[5m]) > 0.05
    for: 5m
    annotations:
      summary: "Error rate above 5%"

  - alert: EvolutionQueueBacklog
    expr: evolution_queue_depth > 100
    for: 10m
    annotations:
      summary: "Evolution queue backlog"

  - alert: LLMAPIRateLimit
    expr: openai_rate_limit_errors > 0
    annotations:
      summary: "OpenAI rate limit hit"
```

---

## 16. Scaling Strategy

### 16.1 Horizontal Scaling

**Screenshot Renderer**:
- Base: 5 replicas
- Scale up trigger: CPU > 70%
- Scale up to: 50 replicas
- Each replica: 10 concurrent renders

**Evolution Engine**:
- Base: 3 replicas
- Scale up trigger: Queue depth > 50
- Scale up to: 20 replicas

**LLM Judge Service**:
- Base: 2 replicas
- Scale up trigger: API latency > 5s
- Scale up to: 10 replicas

### 16.2 Vertical Scaling

**Database**:
- Start: db.t3.medium (2 vCPU, 4GB RAM)
- Scale up to: db.m5.4xlarge (16 vCPU, 64GB RAM)
- Read replicas: 1 → 5

**Redis**:
- Start: cache.t3.medium
- Scale up to: cache.m5.2xlarge
- Redis Cluster for sharding

---

## 17. Cost Management

### 17.1 Cost Breakdown (Per Evolution)

| Component            | Cost (50 pop × 10 gen) |
|----------------------|------------------------|
| Screenshots          | $0 (free with Puppeteer) |
| LLM Judges (4 agents) | $16.25 (500 designs × $0.0325) |
| Storage (R2)         | $0.02 (500 screenshots × 1MB × $0.015/GB) |
| Database             | $0.01 (500 rows) |
| **Total**            | **~$16.28**            |

### 17.2 Pricing Strategy

**Free Tier**:
- 5 evolutions/mo
- Max 50 designs/evolution
- Basic judges only
- Cost to us: ~$81/mo

**Pro Tier ($99/mo)**:
- Unlimited evolutions
- Max 200 designs/evolution
- All 4 judges
- Break-even: ~6 evolutions/mo

**Agency Tier ($499/mo)**:
- Everything in Pro
- Max 500 designs/evolution
- White-label
- Custom judges
- Break-even: ~30 evolutions/mo

**Enterprise**:
- Custom pricing
- Private deployment
- Fine-tuned judges
- $50k+/yr

### 17.3 Cost Optimization

1. **Screenshot Deduplication**:
   - Hash HTML, reuse screenshots
   - Save ~30% on LLM calls

2. **Adaptive Evaluation**:
   - Quick filter with Haiku ($0.001)
   - Full eval only for top 50%
   - Save ~50% on LLM costs

3. **Smart Caching**:
   - Cache judge responses
   - Same screenshot + criteria = cache hit
   - Save ~20% on repeat evaluations

---

## 18. Risk Mitigation

### 18.1 Technical Risks

| Risk                    | Impact | Mitigation |
|-------------------------|--------|------------|
| LLM API rate limits     | High   | Multiple providers, exponential backoff, queue management |
| Screenshot bottlenecks  | Medium | Horizontal scaling, caching, CDN |
| Evolution convergence   | Low    | Adaptive mutation rates, novelty injection |
| Cost overruns           | High   | Budget limits, cost optimization, stop-loss |
| Data loss               | Medium | Database backups, replication, S3 versioning |

### 18.2 Business Risks

| Risk                    | Impact | Mitigation |
|-------------------------|--------|------------|
| Competitors copy idea   | High   | Fast execution, brand moat, data network effects |
| Users don't trust AI    | Medium | Explainable AI, evolution transparency, human-in-loop |
| LLM costs too high      | High   | Cost optimization, bring your own API key, tiered pricing |
| Platform dependency     | Low    | Multi-provider support, open-source fallbacks |

### 18.3 Legal Risks

| Risk                    | Impact | Mitigation |
|-------------------------|--------|------------|
| Copyright infringement  | Medium | User warrants ownership, no training without consent |
| Data privacy violations | High   | GDPR compliance, encryption, right to deletion |
| API ToS violations      | Medium | Review all provider ToS, implement rate limits |

---

## 19. Full Implementation Addenda

This section captures the implementation requirements needed for a complete product launch (beyond the core evolution engine), and maps to the post-MVP work in the todo list.

### 19.1 Credits, Billing & Tier Enforcement
- Maintain a credit ledger tied to users and plans (credit balance, deductions, refunds).
- Enforce tier limits (evolutions per period, population size, max generations).
- Integrate a payment provider for upgrades and renewals (Stripe or equivalent).
- Process billing webhooks to sync plan status and credit allocations.
- Apply budget caps and refund policies for failed evolutions.

### 19.2 Exports & Design System Assets
- Export winning designs as HTML/CSS/assets with metadata.
- Extract design tokens (color, typography, spacing) and expose a component library view.
- Provide A/B testing export formats for external experimentation tools.
- Version and rollback design system exports for reuse across projects.

### 19.3 Feedback, Collaboration & Governance
- Capture explicit feedback (ratings, comments) to refine criteria and judges.
- Support collaboration (teams, invites, roles, shared evolutions).
- Maintain audit logs for sensitive actions (export, delete, admin changes).
- Implement retention and deletion flows aligned with privacy policies.

### 19.4 Enterprise Readiness
- Provide SSO (SAML/OIDC) and SCIM provisioning.
- Allow custom judges and weighted scoring configurations per tenant.
- Enable white-labeling (themes, custom domains, branding assets).
- Expose API access with key management and usage analytics.

### 19.5 Operational Readiness
- Define SLA targets, incident response runbooks, and escalation paths.
- Establish monitoring dashboards with launch-day and post-launch watchlists.
- Provide compliance reporting, audit export, and privacy documentation.

---

## 20. Production Readiness

This section defines the additional requirements to operate the platform safely at scale with predictable reliability, security, and support.

### 20.1 Reliability & SLOs
- Define SLIs/SLOs (latency, error rate, queue depth) with error budgets.
- Set RTO/RPO targets and validate them with backup/restore drills.
- Add capacity planning for renderer/judge throughput and DB growth.
- Establish release gating based on error budget burn.

### 20.2 Security Hardening
- Deploy WAF and DDoS protections for public endpoints.
- Implement secrets management and rotation schedules (KMS or equivalent).
- Add SAST/DAST/SCA scans in CI and enforce high-severity gates.
- Configure image scanning and dependency license checks.
- Perform threat modeling and abuse-case review for core flows and cost controls.
- Require MFA and least-privilege access with periodic access reviews.
- Schedule external penetration tests and remediate high/critical findings.
- Generate SBOMs and sign release artifacts to harden the supply chain.

### 20.3 Data Governance & Privacy
- Formalize data retention, deletion, and export workflows.
- Ensure audit logs are tamper-evident and access-controlled.
- Document data classification and access policies by role/tenant.
- Validate encryption in transit and at rest for screenshots and metadata.

### 20.4 Release Management
- Use feature flags for staged rollout of high-cost features.
- Support blue/green or canary deploys with fast rollback paths.
- Maintain a change log and release checklist for each deployment.

### 20.5 Support & Incident Response
- Stand up a status page with incident templates.
- Define on-call rotation and escalation paths.
- Create runbooks for renderer/judge outages, cost spikes, and queue backlogs.
- Add post-incident review and action tracking.

### 20.6 Compliance & Legal
- Publish privacy policy, terms of service, cookie policy, and DPA (data processing addendum).
- Perform vendor risk review for LLM and storage providers.
- Document compliance controls (GDPR, SOC2 readiness) and evidence capture.     
- Establish vulnerability disclosure policy and security contact.              

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building the Web Design Evolution Platform on the **BubbleLab** and **OpenEvolve** foundation. The architecture is designed to be:

- **Platform-Based**: Built on proven BubbleLab (React/TypeScript) + OpenEvolve (Python/ML) platforms
- **Scalable**: Horizontal and vertical scaling paths
- **Cost-effective**: ~47% reduction in development effort, optimized LLM costs
- **Reliable**: Checkpoints, retries, and error handling
- **Fast**: Parallel processing and caching, leveraging existing infrastructure
- **Secure**: Authentication, authorization, and data encryption
- **Observable**: Comprehensive monitoring and logging

**Platform Foundation:**
- 🟢 **BubbleLab**: 284 existing components (React 19, Vite, Hono, Drizzle ORM, Clerk auth, @xyflow/react)
- 🔧 **OpenEvolve**: 65 existing components (mutation engine, decomposition workflow, MCTS/MDAP algorithms, adversarial/team/gauntlet system)
- **New Development**: 470 tasks (vs. 883 without platforms)
- **Time Savings**: TBD (sync with todo list estimates)

The next step is to begin execution with the **7-day sprint to MVP**:
1. Day 1: Clone repositories, install dependencies (Node.js + Python)
2. Day 2: Integrate OpenEvolve mutation engine, extend AIAgentBubble
3. Day 3: Wire evolution pipeline (TypeScript orchestrator + Python mutations)
4. Day 4: Build mitosis animation using @xyflow/react
5. Day 5: Integration testing + polish
6. Day 6: Record demo video
7. Day 7: Launch!

Let's build this. 🚀


---
## STATUS (Reconciliation Note)
**Last reconciled: 2026-08-20**

- TYPE: Implementation plan for the web-design evolution platform.
- VERIFICATION: grep for 'WebDesign'/'web design'/'design evolution' in core-projects/BubbleLab/apps/bubble-studio/src = 0 matches. Generic evolution UI exists; no web-design-specific implementation (visual LLM judges, HTML/CSS mutation engine, screenshot renderer) found.
- STATUS: DESIGN-ONLY.

