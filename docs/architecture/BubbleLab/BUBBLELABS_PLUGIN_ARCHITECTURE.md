# BubbleLabs Plugin Architecture & Development Guide

This document defines the technical standards and architectural patterns for creating and integrating plugins into the BubbleLabs ecosystem. It is based on the production-grade integration of the **OpenEvolve PyGraphistry Plugin**.

---

## 1. Core Philosophy: The Symmetric Adapter Pattern

BubbleLabs utilizes a **Symmetric Adapter Pattern**. Every plugin is split into two specialized halves that communicate over a standardized REST API:

1.  **Frontend Adapter (TypeScript/React):** Manages the user interface, reactive state, and user-provided configurations (like API keys).
2.  **Backend Adapter (Python/FastAPI):** Orchestrates heavy-duty analytical engines, GPU-accelerated libraries, and local source-code submodules.

This separation ensures that the UI remains lightweight and portable while the backend can leverage the full power of the Python scientific stack.

---

## 2. Technical Architecture

### A. The Naming Convention Bridge
A significant hurdle in cross-language plugins is the naming convention gap.
*   **Frontend (JS/TS):** Standardizes on `camelCase` (e.g., `gpuAcceleration`).
*   **Backend (Python):** Standardizes on `snake_case` (e.g., `gpu_acceleration`).

**Architectural Solution:** The BubbleLabs `IntegrationRegistry` includes an automatic **Normalization Helper**. It recursively converts all incoming JSON configuration keys from `camelCase` to `snake_case` before they reach the backend adapter. Developers should write native code in both languages and let the registry bridge the gap.

### B. Robust Dependency Resolution
Plugins often rely on external projects that may be present as local clones rather than installed `pip` packages. Backend adapters must implement a multi-path resolution strategy:
1.  Check if the module is already in `sys.modules`.
2.  Search the `integrations/` root.
3.  Search known "projects" or "submodules" directories.
4.  Add the verified path to `sys.path` dynamically.

---

## 3. Frontend Plugin Structure

Every plugin should be a self-contained NPM package (e.g., under `@openevolve/*`) with the following internal structure:

```text
/src/
├── types/
│   └── plugin-types.ts     # Interfaces for State, Config, and API Results
├── utils/
│   └── createPlugin.ts     # The "Brain": Singleton manager & fetch client
├── components/
│   ├── VizSettings.tsx     # The "Activation UI": Configuration toggles
│   └── MyComponent.tsx     # The "Display UI": Main visualization
└── index.ts                # Barrel export for all public components
```

### State Management Guidelines
*   **Singleton Pattern:** Use a single exported instance (e.g., `pygraphistryPlugin`) to maintain state consistency across multiple UI components.
*   **Disabled by Default:** All features must initialize as `false`. Users must manually opt-in through the settings panel.
*   **Dynamic Config Pass-through:** Every API request should bundle the current `globalState.config` to allow real-time credential updates without restarting the backend.

---

## 4. Backend Adapter Structure

Backend adapters must reside in `integrations/` and register with the `IntegrationRegistry`.

### A. The Adapter Class
Adapters should inherit from a Base Interface (e.g., `VisualizationInterface`) and implement:
*   `__init__()`: Must support **parameterless instantiation** to allow the registry to discover capabilities before configuration is available.
*   `initialize(config)`: Handles the actual connection to engines using normalized snake_case settings.
*   `validate()`: Performs a self-health check (e.g., checking if GPU drivers are present).

### B. API Route Standards
All routes in `openevolve_api.py` must follow the production standard:
*   **Prefix:** `/api/openevolve/...`
*   **Methodology:** Use `POST` for actions requiring data (like discovery) and `GET` for simple status or summary retrievals.
*   **No Mocks:** Production routes must never return hardcoded "demo" data. If a service is unavailable, return a `503 Service Unavailable` with a descriptive detail.

---

## 5. Attachment Points

### A. The Activation UI
Plugins attach to the user workflow through the **Settings Panel**. This component should:
1.  Provide high-level toggles for each submodule.
2.  Provide input fields for required credentials.
3.  Directly update the Plugin Singleton state.

### B. The Visualization Hook
Primary visualizations should be designed to mount within the **Consolidated Side Panel** of the `bubble-studio` IDE. They consume the `nodes` and `edges` exported by the active `BubbleFlow` and transform them into interactive insights via the backend.

---

## 6. Development Checklist

- [ ] **Type Safety:** `npx tsc --noEmit` passes in the plugin directory.
- [ ] **Prefixing:** All backend routes start with `/api/openevolve`.
- [ ] **Normalization:** Frontend uses `camelCase`, Backend uses `snake_case`.
- [x] **Zero Logic Simulation:** All `setTimeout` and `mockResult` placeholders are replaced with `fetch` calls. Remaining `setTimeout` uses are legitimate `AbortController` request timeouts; `generateMockResult` is retained only as a test utility. The one backend-backed operation that previously returned a hardcoded list (`openevolve-leanaide` `get_models`) now issues a real `GET /api/openevolve/leanaide/models` fetch, with a clearly-marked `offline: true` fallback used only when the OpenEvolve server is unreachable.
- [ ] **Default State:** All features are `disabled` until the user manually toggles them.
- [ ] **Error Handling:** Backend returns `HTTPException` instead of demo data.
- [ ] **Registry Registration:** The plugin is added to `builtin_integrations` in `registry.py`.

---

## 7. Build and Compilation
Plugins are compiled using **Vite in Library Mode**. This produces a highly optimized UMD/ES bundle that can be imported by the main BubbleLabs application without polluting the global namespace.

```bash
# To build for production
npm run build

# To check types
npx tsc --noEmit
```
