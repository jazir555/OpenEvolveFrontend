# LeanAIDE BubbleLab Plugin Integration Example

This document shows how to integrate the LeanAIDE plugin into BubbleLab without modifying the core BubbleLab codebase.

## Integration Steps

### 1. Install the Plugin

```bash
# Install the plugin package
npm install leanaide-bubblelab-plugin

# Or yarn
yarn add leanaide-bubblelab-plugin
```

### 2. Initialize the Plugin

In your BubbleLab application's main entry point (e.g., `main.tsx` or `App.tsx`):

```typescript
import { initializeLeanAideClient, initializeRagbitsClient } from 'leanaide-bubblelab-plugin';

// Initialize LeanAIDE client during app startup
initializeLeanAideClient({
  serverUrl: import.meta.env.VITE_LEANAIDE_SERVER_URL || 'http://localhost:3000/leanaide',
  apiKey: import.meta.env.VITE_LEANAIDE_API_KEY
});

initializeRagbitsClient({
  serverUrl: import.meta.env.VITE_RAGBITS_SERVER_URL || 'http://localhost:3000/ragbits',
  apiKey: import.meta.env.VITE_RAGBITS_API_KEY
});
```

### 3. Add Environment Variables

```env
# .env
VITE_LEANAIDE_SERVER_URL=https://your-leanaide-server.com/api
VITE_LEANAIDE_API_KEY=your-api-key
VITE_RAGBITS_SERVER_URL=https://your-ragbits-server.com/api
VITE_RAGBITS_API_KEY=your-api-key
```

### 4. Use in BubbleLab Components

#### Option A: Using the React Component

```typescript
import { LeanAideVerification } from 'leanaide-bubblelab-plugin';
import { RagbitsKnowledgeSearch } from 'leanaide-bubblelab-plugin';

function EnhancedBubbleLabComponent() {
  const [verificationResult, setVerificationResult] = useState(null);
  
  return (
    <div className="bubblelab-component">
      {/* Your existing BubbleLab UI */}
      
      <div className="verification-section">
        <LeanAideVerification
          problemStatement="Prove that the sum of two even numbers is even"
          solutionCode="theorem even_plus_even : ∀ a b : ℕ, even a → even b → even (a + b)"
          mode="verification"
          onVerificationResult={(result) => {
            console.log('LeanAIDE verification result:', result);
            setVerificationResult(result);
          }}
        />
      </div>

      <div className="knowledge-search-section">
        <RagbitsKnowledgeSearch
          initialQuery="authentication patterns"
          topK={5}
          onResults={(results) => {
            console.log('RAGBits search results:', results);
          }}
        />
      </div>
    </div>
  );
}
```

#### Option B: Using the Plugin API

```typescript
import { LeanAidePlugin } from 'leanaide-bubblelab-plugin';

async function verifyMathematicalSolution(problem: string, solution: string) {
  try {
    const result = await LeanAidePlugin.verifySolution(problem, solution);
    
    if (result.success) {
      console.log('Verification successful:', result.data);
      return { success: true, data: result.data };
    } else {
      console.error('Verification failed:', result.error);
      return { success: false, error: result.error };
    }
  } catch (error) {
    console.error('LeanAIDE error:', error);
    return { success: false, error: 'LeanAIDE service error' };
  }
}
```

### 5. Using the Panel Component

```typescript
import { LeanAidePanel } from 'leanaide-bubblelab-plugin';

function BubbleLabWorkflowWithLeanAide() {
  const [showLeanAidePanel, setShowLeanAidePanel] = useState(false);
  const [currentProblem, setCurrentProblem] = useState('');
  const [currentSolution, setCurrentSolution] = useState('');
  
  return (
    <div className="workflow-container">
      {/* Your existing workflow UI */}
      
      <button 
        onClick={() => setShowLeanAidePanel(!showLeanAidePanel)}
        className="lean-aide-toggle"
      >
        {showLeanAidePanel ? 'Hide LeanAIDE' : 'Show LeanAIDE'}
      </button>
      
      {showLeanAidePanel && (
        <div className="lean-aide-panel-container">
          <LeanAidePanel
            isOpen={showLeanAidePanel}
            onClose={() => setShowLeanAidePanel(false)}
            problemStatement={currentProblem}
            solutionCode={currentSolution}
            onVerificationResult={(result) => {
              console.log('LeanAIDE verification result:', result);
              // Handle the result in your workflow
            }}
          />
        </div>
      )}
    </div>
  );
}
```

## Advanced Integration

### Customizing the Plugin

You can customize the plugin's appearance and behavior:

```typescript
import { LeanAideVerification } from 'leanaide-bubblelab-plugin';

function CustomLeanAideIntegration() {
  return (
    <div className="custom-lean-aide-container">
      <LeanAideVerification
        problemStatement="Your mathematical problem"
        solutionCode="Your solution code"
        mode="verification"
        className="custom-styles"
        onVerificationResult={(result) => {
          // Custom result handling
          if (result.success) {
            // Show success notification
          } else {
            // Show error notification
          }
        }}
      />
    </div>
  );
}
```

### Using Different Modes

The plugin supports multiple operation modes:

```typescript
import { LeanAideVerification } from 'leanaide-bubblelab-plugin';

function LeanAideModesDemo() {
  return (
    <div className="modes-demo">
      <h3>Theorem Translation</h3>
      <LeanAideVerification
        problemStatement="For all natural numbers n, n + 0 = n"
        mode="theorem"
      />

      <h3>Definition Translation</h3>
      <LeanAideVerification
        problemStatement="A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself"
        mode="definition"
      />

      <h3>Math Query</h3>
      <LeanAideVerification
        problemStatement="What is the sum of the first 100 natural numbers?"
        mode="query"
      />
    </div>
  );
}
```

## Error Handling

The plugin provides comprehensive error handling:

```typescript
import { LeanAidePlugin, isLeanAideAvailable } from 'leanaide-bubblelab-plugin';

async function safeLeanAideOperation() {
  // Check if LeanAIDE is available
  if (!isLeanAideAvailable()) {
    console.warn('LeanAIDE service is not available');
    return;
  }

  try {
    const result = await LeanAidePlugin.verifySolution(
      'Prove that 2 + 2 = 4',
      'theorem two_plus_two : 2 + 2 = 4 := rfl'
    );

    if (result.success) {
      console.log('Success:', result.data);
    } else {
      console.error('Failed:', result.error);
    }
  } catch (error) {
    console.error('Network error:', error);
  }
}
```

## Integration Benefits

1. **No Core Modifications**: The plugin doesn't require changes to BubbleLab's core codebase
2. **Clean Separation**: LeanAIDE functionality is isolated in a separate package
3. **Easy Updates**: The plugin can be updated independently of BubbleLab
4. **Type Safety**: Full TypeScript support with type definitions
5. **Flexible Usage**: Can be used as React components or via the plugin API

## Troubleshooting

### LeanAIDE Service Not Available

If you see "LeanAIDE service is not available" errors:

1. Check that you've called `initializeLeanAideClient()` during app startup
2. Verify that the server URL is correct
3. Ensure the LeanAIDE server is running and accessible

### Network Errors

If you encounter network errors:

1. Check your network connection
2. Verify the LeanAIDE server URL
3. Ensure CORS is properly configured on the LeanAIDE server
4. Check that the API key is valid (if required)

### TypeScript Errors

If you have TypeScript compilation errors:

1. Ensure you're using compatible versions of React and TypeScript
2. Check that all peer dependencies are installed
3. Verify that the plugin version is compatible with your BubbleLab version

## Support

For issues with the plugin integration, please open an issue on the GitHub repository with details about your BubbleLab version and the specific problem you're encountering.
