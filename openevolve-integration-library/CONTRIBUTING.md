# Contributing to OpenEvolve Integration Library

Thank you for your interest in contributing! This document provides guidelines for contributing to the OpenEvolve Integration Library.

## Project Structure (v1.1.0+)

```
openevolve-integration-library/
├── src/
│   ├── api/              # API client, backend communication, middleware
│   ├── integrations/     # Integration implementations and base adapter
│   ├── react/            # React hooks and provider
│   ├── store/            # Zustand state management
│   ├── testing/          # Mocking utilities
│   ├── utils/            # Shared utility functions
│   └── index.ts          # Main library entry point
├── examples/             # Usage examples (TS/TSX)
├── dist/                 # Build output (generated)
└── package.json
```

## Adding a New Integration

To add a new integration:

1. **Implement the adapter in `src/integrations/all-integrations.ts`**
   - Extend `BaseIntegrationAdapter`.
   - Define `Inputs` and `Result` interfaces.
   - Implement `execute`, `validate`, and `getSchema`.
   - Add convenience methods.

2. **Register in `OpenEvolveClient` (`src/api/client.ts`)**
   - Add the integration to `IntegrationName` enum.
   - Add to `IntegrationRegistry` interface.
   - Instantiate in the `loadIntegrations` method.

3. **Export Types**
   - Export new interfaces from `src/integrations/index.ts`.

4. **Add React Hook (Optional)**
   - Add a specialized hook in `src/react/index.ts`.

## Code Style

- **Naming**: Use `snake_case` for all backend-facing parameters and properties. Use `camelCase` for internal class methods and variables.
- **Strict Typing**: Avoid `any`. Every integration must have explicit Input and Result interfaces.
- **Validation**: Use base class helpers (`validateRequired`, `validateEnum`) in your `validate` method.

## Testing

- Write unit tests in `src/tests/`.
- Ensure `npm test` passes before submitting PRs.
- Use `createMockClient` from the library's own testing utility for integration tests.

## Commit Messages

Follow conventional commit format:
`type(scope): description`

Example: `feat(maker): add support for python-3.11 runtime`