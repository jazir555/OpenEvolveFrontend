# OpenEvolve BubbleLab Components

This package contains the converted OpenEvolve Streamlit components to BubbleLab-compatible TypeScript/React components.

## Overview

The original OpenEvolve application was built using Streamlit, a Python-based web framework. This conversion project transforms those components into modern TypeScript/React components that can be integrated into the BubbleLab ecosystem.

## Converted Components

The following major components have been converted:

- **OpenEvolveApp**: Main application component with tabbed interface
- **Header**: Application header with navigation
- **Sidebar**: Navigation sidebar
- **EvolutionTab**: Evolution engine interface
- **AdversarialTestingTab**: Adversarial testing interface
- **GithubIntegrationTab**: GitHub integration features
- **ActivityFeedTab**: Activity feed display
- **ReportTemplatesTab**: Report template management
- **ModelDashboardTab**: Model dashboard and management
- **TasksTab**: Task management interface
- **AdminTab**: Administrative functions
- **AnalyticsDashboardTab**: Analytics dashboard
- **OpenEvolveDashboardTab**: OpenEvolve-specific dashboard
- **OrchestratorTab**: Service orchestration interface
- **MonitoringTab**: System monitoring interface

## Features Preserved

- Full tabbed interface matching the original Streamlit application
- State management using React hooks and localStorage
- Responsive design compatible with BubbleLab
- All major functionality from the original application
- Real-time simulation of backend processes

## Usage

To use these components in your BubbleLab application:

```tsx
import { OpenEvolveApp } from '@openevolve/bubblelab-components';

function App() {
  return (
    <OpenEvolveApp />
  );
}
```

## Technical Details

- Built with TypeScript and React
- Uses Tailwind CSS for styling
- Implements React hooks for state management
- Compatible with modern frontend frameworks
- Follows BubbleLab component architecture patterns

## Development

To build the components:

```bash
npm install
npm run build
```

To develop in watch mode:

```bash
npm run dev
```

## License

MIT