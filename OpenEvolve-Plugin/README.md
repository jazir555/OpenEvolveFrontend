# OpenEvolve Plugin

> AI Evolution and Optimization Platform for BubbleLab

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/openevolve/plugin)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue.svg)](https://www.typescriptlang.org/)

## Overview

OpenEvolve Plugin is a comprehensive AI evolution and optimization platform designed for the BubbleLab studio ecosystem. It provides advanced workflow capabilities, analytics, knowledge base management, and integration with LeanAide for formal verification.

## Features

### Core Capabilities

- **Evolutionary Optimization**: Advanced genetic algorithms and evolutionary strategies
- **Adversarial Testing**: Robust adversarial attack and defense capabilities
- **MDAP Integration**: Multi-Decision Process Agent for complex decision-making
- **Problem Decomposition**: Intelligent problem breaking and parallel solving
- **Knowledge Base**: Vector-based semantic search and artifact management
- **LeanAide Integration**: Formal proof verification and generation

### Workflow Support

- **10 Workflow Types**: Evolution, Adversarial, Maker, MDAP, Decomposition, Knowledge, LeanAide, CrewAI, ROMA, Invention
- **Real-time Monitoring**: Live execution tracking with WebSocket support
- **Visual Analytics**: Comprehensive performance metrics and visualizations
- **Configurable Parameters**: 272 configurable parameters across all workflows

### Integration Services

- **CrewAI**: Advanced task delegation and resource management
- **ROMA**: Multi-objective optimization with Pareto fronts
- **End-to-End Invention**: Automated invention pipeline with evaluation

## Installation

### As a BubbleLab Plugin

```bash
cd BubbleLab/apps/bubble-studio
npm install @openevolve/plugin
```

### Standalone Usage

```bash
cd OpenEvolve-Plugin
npm install
npm run build
```

## Quick Start

```typescript
import { OpenEvolvePlugin, WorkflowBuilder } from '@openevolve/plugin';

// Initialize the plugin
const plugin = OpenEvolvePlugin;

// Create a workflow
const workflow = {
  name: 'My Evolution Workflow',
  type: 'evolution',
  config: {
    population_size: 100,
    generations: 50,
    mutation_rate: 0.1,
  },
};

// Execute workflow
const result = await plugin.services.evolution.execute(workflow);
```

## Documentation

- [Architecture](./docs/ARCHITECTURE.md)
- [API Reference](./docs/API.md)
- [Workflow Guide](./docs/WORKFLOWS.md)
- [Configuration](./docs/CONFIGURATION.md)
- [Integration Guide](./docs/INTEGRATION.md)

## Project Structure

```
OpenEvolve-Plugin/
├── src/
│   ├── components/        # React components (26 total)
│   ├── services/          # API clients and hooks
│   ├── stores/            # Zustand state stores
│   ├── schemas/           # Zod validation schemas
│   ├── types/             # TypeScript definitions
│   ├── utils/             # Utility functions
│   └── assets/            # Icons and images
├── tests/                 # Test suites
└── dist/                  # Build output
```

## Component Exports

### Pages
- `OpenEvolveDashboard`
- `AnalyticsDashboard`
- `WorkflowBuilder`
- `LeanAidePage`
- `KnowledgeBasePage`

### Workflow Components
- `WorkflowCard`
- `WorkflowList`
- `ExecutionMonitor`
- `ConfigPanel`
- `WorkflowTabs`

### Analytics Components
- `MetricCard`
- `PerformanceChart`
- `ArtifactTable`
- `StatGrid`

### Knowledge Components
- `ArtifactList`
- `KnowledgeSearch`
- `ArtifactEditor`
- `ArtifactDetail`

### LeanAide Components
- `ProofEditor`
- `ModelSelector`
- `VerificationDisplay`
- `ProgressTracker`

### Shared Components
- `ProgressBar`
- `LiveLogViewer`
- `FormWrapper`
- `StatusBadge`

## Development

```bash
# Install dependencies
npm install

# Development mode (watch build)
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run tests with UI
npm run test:ui

# Lint code
npm run lint

# Format code
npm run format
```

## Configuration

Each workflow type has its own configuration schema defined in `src/schemas/`:

- `evolution.ts` - Evolutionary algorithms
- `adversarial.ts` - Adversarial attacks/defenses
- `maker.ts` - MDP Maker workflows
- `mdap.ts` - Multi-Decision Process Agents
- `decomposition.ts` - Problem decomposition
- `knowledge.ts` - Knowledge base operations
- `leanaide.ts` - Lean proof verification
- `crewai.ts` - Task delegation
- `roma.ts` - Multi-objective optimization
- `invention.ts` - Invention pipeline

## API Endpoints

The plugin provides the following API endpoints:

- `POST /api/openevolve/workflows` - Create workflow
- `GET /api/openevolve/workflows` - List workflows
- `GET /api/openevolve/workflows/:id` - Get workflow details
- `POST /api/openevolve/workflows/:id/execute` - Execute workflow
- `GET /api/openevolve/analytics` - Get analytics data
- `GET /api/openevolve/knowledge/search` - Search knowledge base
- `POST /api/openevolve/leanaide/verify` - Verify Lean proof

## WebSocket Events

Connect to `ws://localhost/ws/openevolve` for real-time updates:

- `workflow.started` - Workflow execution started
- `workflow.updated` - Workflow progress update
- `workflow.completed` - Workflow execution completed
- `workflow.failed` - Workflow execution failed
- `log.message` - New log message
- `proof.progress` - Proof verification progress

## State Management

The plugin uses Zustand for state management with the following stores:

- `useAuthStore` - Authentication state
- `useWorkflowStore` - Workflow state
- `useAnalyticsStore` - Analytics state
- `useKnowledgeStore` - Knowledge base state
- `useLeanAideStore` - LeanAide state
- `useEvolutionStore` - Evolution state

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

OpenEvolve Team

## Support

- Documentation: [https://docs.openevolve.ai](https://docs.openevolve.ai)
- Issues: [https://github.com/openevolve/plugin/issues](https://github.com/openevolve/plugin/issues)
- Discussions: [https://github.com/openevolve/plugin/discussions](https://github.com/openevolve/plugin/discussions)

## Acknowledgments

- Built with [React](https://react.dev)
- State management by [Zustand](https://zustand-demo.pmnd.rs)
- Data fetching with [TanStack Query](https://tanstack.com/query/latest)
- Form validation with [Zod](https://zod.dev)

---

Made with ❤️ by the OpenEvolve Team
