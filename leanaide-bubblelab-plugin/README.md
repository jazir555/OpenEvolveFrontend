# LeanAIDE BubbleLab Plugin

A standalone plugin that integrates LeanAIDE formal verification capabilities into BubbleLab workflows without modifying the core BubbleLab codebase.

## Overview

This plugin provides LeanAIDE integration as a standalone package that can be imported and used by BubbleLab without requiring modifications to the main BubbleLab repository.

## Features

- **Standalone Integration**: Works as a separate package that BubbleLab can import
- **LeanAIDE Client**: TypeScript client for LeanAIDE API
- **React Components**: Pre-built UI components for LeanAIDE verification
- **Service Layer**: Clean abstraction for LeanAIDE operations
- **RAGBits Knowledge Search**: Semantic search UI and client for workflow artifacts
- **No Core Modifications**: Doesn't require changes to BubbleLab's main codebase

## Installation

```bash
# Install the plugin package
npm install leanaide-bubblelab-plugin

# Or yarn
yarn add leanaide-bubblelab-plugin
```

## Usage

### Basic Integration

```typescript
import { LeanAidePlugin, RagbitsPlugin } from 'leanaide-bubblelab-plugin';
import { initializeLeanAideClient, initializeRagbitsClient } from 'leanaide-bubblelab-plugin';

// Initialize the client
initializeLeanAideClient({
  serverUrl: 'https://your-leanaide-server.com/api',
  apiKey: 'your-api-key'
});

initializeRagbitsClient({
  serverUrl: 'https://your-ragbits-server.com/api',
  apiKey: 'your-api-key'
});

// Use in your BubbleLab components
function MyBubbleLabComponent() {
  const [verificationResult, setVerificationResult] = useState(null);
  
  const handleVerify = async (problemStatement, solutionCode) => {
    const result = await LeanAidePlugin.verifySolution(problemStatement, solutionCode);
    setVerificationResult(result);
  };
  
  return (
    <div>
      {/* Your existing BubbleLab UI */}
      <LeanAideVerificationPanel 
        problemStatement="Your mathematical problem"
        solutionCode="Your solution code"
        onVerificationResult={setVerificationResult}
      />
    </div>
  );
}
```

### Using the React Component

```typescript
import { LeanAideVerification } from 'leanaide-bubblelab-plugin';

function MyWorkflowComponent() {
  return (
    <div className="workflow-container">
      {/* Your existing workflow components */}
      
      <div className="verification-panel">
        <LeanAideVerification
          problemStatement="Prove that 1 + 1 = 2"
          solutionCode="theorem one_plus_one : 1 + 1 = 2 := rfl"
          mode="verification"
          onVerificationResult={(result) => {
            console.log('Verification result:', result);
          }}
        />
      </div>
    </div>
  );
}
```

## API Reference

### LeanAideClient

```typescript
interface LeanAideConfig {
  serverUrl: string;
  apiKey?: string;
}

class LeanAideClient {
  constructor(config: LeanAideConfig);
  
  translateTheorem(theoremStatement: string, context?: string): Promise<LeanAideTaskResponse>;
  translateDefinition(definitionStatement: string, context?: string): Promise<LeanAideTaskResponse>;
  verifySolution(problem: string, solution: string, context?: string): Promise<LeanAideTaskResponse>;
  elaborateCode(leanCode: string, context?: string): Promise<LeanAideTaskResponse>;
  mathQuery(query: string, context?: string): Promise<LeanAideTaskResponse>;
}
```

### LeanAideService

```typescript
function getLeanAideClient(): LeanAideClient;
function initializeLeanAideClient(config: { serverUrl?: string; apiKey?: string }): void;
function translateTheorem(theoremStatement: string, context?: string): Promise<LeanAideTaskResponse>;
function translateDefinition(definitionStatement: string, context?: string): Promise<LeanAideTaskResponse>;
function verifySolution(problem: string, solution: string, context?: string): Promise<LeanAideTaskResponse>;
function elaborateCode(leanCode: string, context?: string): Promise<LeanAideTaskResponse>;
function mathQuery(query: string, context?: string): Promise<LeanAideTaskResponse>;
function isLeanAideAvailable(): boolean;
```

### React Components

#### LeanAideVerification

```typescript
interface LeanAideVerificationProps {
  problemStatement: string;
  solutionCode?: string;
  onVerificationResult?: (result: LeanAideTaskResponse) => void;
  mode?: 'theorem' | 'definition' | 'verification' | 'query' | 'elaboration';
}

function LeanAideVerification(props: LeanAideVerificationProps): JSX.Element;
```

#### LeanAidePanel

```typescript
interface LeanAidePanelProps {
  isOpen: boolean;
  onClose: () => void;
  problemStatement: string;
  solutionCode?: string;
  onVerificationResult?: (result: LeanAideTaskResponse) => void;
}

function LeanAidePanel(props: LeanAidePanelProps): JSX.Element;
```

#### RagbitsKnowledgeSearch

```typescript
interface RagbitsKnowledgeSearchProps {
  initialQuery?: string;
  filters?: Record<string, unknown>;
  topK?: number;
  onResults?: (results: RagbitsSearchResult[]) => void;
}

function RagbitsKnowledgeSearch(props: RagbitsKnowledgeSearchProps): JSX.Element;
```

## Configuration

### Environment Variables

```env
# .env
VITE_LEANAIDE_SERVER_URL=https://your-leanaide-server.com/api
VITE_LEANAIDE_API_KEY=your-api-key
VITE_RAGBITS_SERVER_URL=https://your-ragbits-server.com/api
VITE_RAGBITS_API_KEY=your-api-key
```

### Initialization

```typescript
import { initializeLeanAideClient, initializeRagbitsClient } from 'leanaide-bubblelab-plugin';

// Initialize during app startup
initializeLeanAideClient({
  serverUrl: import.meta.env.VITE_LEANAIDE_SERVER_URL,
  apiKey: import.meta.env.VITE_LEANAIDE_API_KEY
});

initializeRagbitsClient({
  serverUrl: import.meta.env.VITE_RAGBITS_SERVER_URL,
  apiKey: import.meta.env.VITE_RAGBITS_API_KEY
});
```

## Development

### Building the Plugin

```bash
# Install dependencies
npm install

# Build the plugin
npm run build

# Run tests
npm test
```

### Project Structure

```
leanaide-bubblelab-plugin/
├── src/
│   ├── lib/
│   │   └── leanaideClient.ts      # Core client implementation
│   │   └── ragbitsClient.ts       # RAGBits client implementation
│   ├── services/
│   │   └── leanaideService.ts     # Service layer
│   │   └── ragbitsService.ts      # RAGBits service layer
│   ├── components/
│   │   ├── LeanAideVerification.tsx # Main verification component
│   │   └── LeanAidePanel.tsx      # Panel component
│   │   └── RagbitsKnowledgeSearch.tsx # Knowledge search component
│   └── index.ts                   # Main export
├── public/
│   └── leanaide.svg              # Plugin logo
├── package.json
├── tsconfig.json
└── README.md
```

## Integration with BubbleLab

This plugin is designed to be imported and used by BubbleLab without requiring any modifications to BubbleLab's core codebase. The plugin provides:

1. **Self-contained Components**: React components that can be imported and used directly
2. **Clean API**: Well-defined TypeScript interfaces for easy integration
3. **No Side Effects**: Doesn't modify global state or BubbleLab's internal structures
4. **Easy Configuration**: Simple initialization and configuration

## License

MIT License

## Support

For issues and feature requests, please open an issue on the GitHub repository.
