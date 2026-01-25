# Changelog

All notable changes to the OpenEvolve Integration Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-01-09

### Added
- **Middleware System**: Added support for custom execution pipelines via `Middleware` interface.
- **React Support**: Added `OpenEvolveProvider` and specialized hooks (`useDecomposition`, `useLeanAide`, etc.).
- **State Management**: Added Zustand store factory (`createOpenEvolveStore`) for robust state tracking.
- **New Integrations**: Fully implemented `Solution` and `Assembly` integration adapters.
- **Testing Utilities**: Added `createMockClient` for easier unit testing of consumer applications.
- **Improved Validation**: Hardened all 9 integration adapters with strict input validation using base helpers.

### Fixed
- **Integration Loading**: Resolved bug where integration adapters were not being instantiated in the client.
- **Streaming Logic**: Fixed progress callback routing for WebSocket-based execution.
- **Error Mapping**: Improved backend error handling to correctly distinguish timeout and network errors.
- **Type Safety**: Unified `ValidationError` types and resolved numerous naming collisions and generic type issues.
- **Duplicate Code**: Consolidated redundant integration implementations and base classes.

### Changed
- Refactored `OpenEvolveClient.execute` to support the middleware pipeline and standardized retry logic.
- Updated all examples (`basic-usage.ts`, `react-usage.tsx`) to match the refined 1.1.0 API.
- Standardized project structure and exports.

## [1.0.0] - 2025-01-03

### Added
- Initial release of OpenEvolve Integration Library
- Unified API client for all OpenEvolve components
- Base integration class for extensibility
- Implementations for:
  - Decomposition Integration
  - LeanAide Integration (formal verification, MCTS, MDAP)
  - Evolution Integration (evolutionary algorithms, adversarial testing)
  - Knowledge Engine Integration (knowledge graphs, extraction)
  - Maker Engine Integration (tool and workflow creation)
  - Hephaestus Integration (delegation, orchestration)
- TypeScript type definitions for all integrations
- Utility functions for validation, error handling, and data manipulation
- Comprehensive documentation and examples
- Support for streaming execution (where applicable)
- Batch execution support
- Health check and version checking

### Features
- Unified interface across all integrations
- Full TypeScript support with type definitions
- Extensible architecture for adding new integrations
- Input validation with JSON schema
- Structured error handling
- Configuration management
- Debug logging support
- Retry logic and timeout handling
- Progress updates for long-running operations

### Documentation
- Comprehensive README with usage examples
- API documentation for all integrations
- Contributing guidelines
- Type documentation
