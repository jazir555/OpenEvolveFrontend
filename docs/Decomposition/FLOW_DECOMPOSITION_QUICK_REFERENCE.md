# Flow Decomposition Quick Reference Guide

## Overview

Flow decomposition is a feature that transforms raw bubble parameters into a structured, UI-ready format with dependency analysis, validation rules, and metadata.

## API Endpoint

**POST** `/api/bubbleflow-template/data-analyst`

### Request Body
```json
{
  "name": "My Data Analyst Bot",
  "description": "Analyzes user data",
  "roles": "Be a data analyst",
  "useCase": "slack-data-scientist"
}
```

### Response (with Flow Decomposition)
```json
{
  "id": 123,
  "name": "My Data Analyst Bot",
  "description": "Analyzes user data",
  "eventType": "slack/bot_mentioned",
  "displayedBubbleParameters": { ... },
  "flowDecomposition": {
    "displayedParameters": [
      {
        "name": "postgres.connectionString",
        "displayName": "Connection String",
        "value": "process.env.DATABASE_URL",
        "type": "env",
        "isRequired": true,
        "isConfigurable": true,
        "description": "Connection String parameter for postgres",
        "group": "postgres",
        "dependencies": ["process", "env", "DATABASE_URL"],
        "source": "environment"
      }
    ],
    "dependencies": {
      "nodes": [
        {
          "id": "postgres",
          "type": "bubble",
          "label": "postgres"
        }
      ],
      "edges": [
        {
          "from": "postgres",
          "to": "postgres.connectionString",
          "type": "data",
          "description": "contains"
        }
      ]
    },
    "validationRules": [
      {
        "type": "required",
        "message": "Connection String is required",
        "severity": "error"
      }
    ],
    "metadata": {
      "totalParameters": 7,
      "requiredParameters": 7,
      "configurableParameters": 6,
      "environmentParameters": 1,
      "nestedParameterCount": 1,
      "conditionalParameterCount": 0,
      "hasCircularDependencies": false,
      "estimatedComplexity": "simple",
      "groups": [
        {
          "name": "postgres",
          "label": "Postgres",
          "parameters": ["postgres.connectionString", "postgres.query"],
          "description": "Parameters for postgres",
          "order": 0
        }
      ]
    }
  },
  "bubbleParameters": { ... },
  "requiredCredentials": { ... },
  "createdAt": "2026-01-10T...",
  "updatedAt": "2026-01-10T..."
}
```

---

## Flow Decomposition Structure

### `displayedParameters[]`

Array of UI-ready parameter objects:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique parameter ID (e.g., `postgres.query`) |
| `displayName` | string | Human-readable name (e.g., "Query") |
| `value` | unknown | Parameter value |
| `type` | enum | `string`, `number`, `boolean`, `env`, `object`, `array`, `unknown` |
| `isRequired` | boolean | Whether parameter is required |
| `isConfigurable` | boolean | Whether user can modify this parameter |
| `description` | string? | Parameter description |
| `group` | string? | Bubble name this parameter belongs to |
| `dependencies` | string[] | List of parameter/bubble dependencies |
| `source` | enum | `literal`, `reference`, `environment`, `computed` |

### `dependencies`

Graph structure for visualizing bubble and parameter relationships:

#### `nodes[]`

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique node ID |
| `type` | enum | `bubble`, `parameter`, `trigger` |
| `label` | string | Display label |

#### `edges[]`

| Field | Type | Description |
|-------|------|-------------|
| `from` | string | Source node ID |
| `to` | string | Target node ID |
| `type` | enum | `data`, `control`, `resource` |
| `description` | string? | Edge description |

### `validationRules[]`

Array of validation constraints:

| Field | Type | Description |
|-------|------|-------------|
| `type` | enum | `required`, `format`, `range`, `custom` |
| `message` | string | Validation message to display |
| `severity` | enum | `error`, `warning`, `info` |

### `metadata`

Summary statistics and analysis:

| Field | Type | Description |
|-------|------|-------------|
| `totalParameters` | number | Total number of parameters |
| `requiredParameters` | number | Count of required parameters |
| `configurableParameters` | number | Count of user-configurable parameters |
| `environmentParameters` | number | Count of environment variable parameters |
| `nestedParameterCount` | number | Count of object/array parameters |
| `conditionalParameterCount` | number | Count of conditional parameters |
| `hasCircularDependencies` | boolean | Whether circular dependencies exist |
| `estimatedComplexity` | enum | `simple`, `medium`, `complex` |
| `groups[]` | array | Parameter groups by bubble |

---

## Use Cases

### 1. Display Parameters in UI

```typescript
const decomposition = response.flowDecomposition;

decomposition.displayedParameters.forEach(param => {
  console.log(`${param.displayName}: ${param.value}`);
  console.log(`Type: ${param.type}, Required: ${param.isRequired}`);
});
```

### 2. Visualize Flow Dependencies

```typescript
import { Graph } from 'some-graph-library';

const graph = new Graph();
const { nodes, edges } = decomposition.dependencies;

// Add nodes
nodes.forEach(node => graph.addNode(node.id, node));

// Add edges
edges.forEach(edge => graph.addEdge(edge.from, edge.to, edge));

// Render graph
graph.render();
```

### 3. Show Validation Errors

```typescript
decomposition.validationRules
  .filter(rule => rule.severity === 'error')
  .forEach(rule => {
    console.error(`❌ ${rule.message}`);
  });
```

### 4. Group Parameters for Display

```typescript
decomposition.metadata.groups.forEach(group => {
  console.log(`## ${group.label}`);
  group.parameters.forEach(paramName => {
    const param = decomposition.displayedParameters.find(p => p.name === paramName);
    console.log(`- ${param?.displayName}: ${param?.value}`);
  });
});
```

### 5. Check Flow Complexity

```typescript
const { estimatedComplexity, hasCircularDependencies } = decomposition.metadata;

if (estimatedComplexity === 'complex') {
  showWarning('This flow is complex and may require careful configuration');
}

if (hasCircularDependencies) {
  showError('This flow has circular dependencies that may cause issues');
}
```

---

## Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | String literal | `"SELECT * FROM users"` |
| `number` | Numeric value | `"5000"` |
| `boolean` | Boolean value | `"true"` or `"false"` |
| `env` | Environment variable | `"process.env.DATABASE_URL"` |
| `object` | Object literal | `'{"ssl": true}'` |
| `array` | Array literal | `'[{"name": "tool"}]'` |
| `unknown` | Cannot determine type | Complex expression |

## Parameter Sources

| Source | Description | Example |
|--------|-------------|---------|
| `literal` | Hard-coded value | `"https://api.example.com"` |
| `reference` | References another bubble | `"aiAgent.responseText"` |
| `environment` | From environment variable | `"process.env.API_KEY"` |
| `computed` | Calculated value | Function expression |

## Dependency Types

| Type | Description |
|------|-------------|
| `data` | Data flow between bubbles/parameters |
| `control` | Control flow (execution order) |
| `resource` | Resource dependency (e.g., environment) |

## Complexity Levels

| Level | Criteria |
|-------|----------|
| `simple` | ≤ 10 parameters, ≤ 15 edges, no cycles |
| `medium` | 10-20 parameters, 15-30 edges, or > 3 env vars |
| `complex` | > 20 parameters, > 30 edges, or has cycles |

---

## Direct Function Usage

You can also use the decomposition function directly:

```typescript
import { generateDisplayedBubbleParameters } from './services/bubble-flow-parser.js';

const bubbleParameters = {
  postgres: {
    variableName: 'postgres',
    bubbleName: 'postgres',
    className: 'PostgresBubble',
    parameters: [
      {
        name: 'connectionString',
        value: 'process.env.DATABASE_URL',
        type: 'env'
      }
    ],
    hasAwait: true,
    hasActionCall: false
  }
};

const decomposition = generateDisplayedBubbleParameters(bubbleParameters);

console.log(decomposition.displayedParameters);
console.log(decomposition.dependencies);
console.log(decomposition.validationRules);
console.log(decomposition.metadata);
```

---

## Testing

Run the test suite:

```bash
cd BubbleLab/apps/bubblelab-api
npx tsx manual-tests/test-flow-decomposition-runner.ts
```

Run realistic flow test:

```bash
npx tsx manual-tests/test-realistic-flow.ts
```

---

## Files

- **Implementation**: `BubbleLab/apps/bubblelab-api/src/services/bubble-flow-parser.ts`
- **API Route**: `BubbleLab/apps/bubblelab-api/src/routes/bubble-flow-templates.ts`
- **Schema**: `BubbleLab/packages/bubble-shared-schemas/src/generate-bubbleflow-schema.ts`
- **Tests**: `BubbleLab/apps/bubblelab-api/src/test/flow-decomposition.test.ts`

---

## Status: ✅ Production Ready

All tests passing. Fully integrated into API.
