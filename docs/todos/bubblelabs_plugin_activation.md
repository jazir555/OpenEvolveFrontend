# Plugin Activation Guide: OpenEvolve PyGraphistry

To activate the **OpenEvolve PyGraphistry Plugin** within the BubbleLabs environment without modifying the core application, follow this two-tier activation procedure.

---

## 1. Backend Activation (The Analytical Engine)

The backend logic is already registered in the OpenEvolve `IntegrationRegistry`. To make it available:

1.  **Enable the Configuration:**
    Ensure `integrations/pygraphistry/config.yaml` has the project enabled:
    ```yaml
    project:
      name: pygraphistry
      enabled: true
    ```

2.  **Start the API Server:**
    Execute the standardized API from the root directory:
    ```bash
    python openevolve_api.py --port 8002
    ```
    The engine is now active and standardized with the `/api` prefix for all analytical routes.

---

## 2. Frontend Activation (The User Interface)

Since the BubbleLabs core is locked, activation is achieved via **Environment Proxying** and **Runtime Initialization**.

1.  **Configure the Network Bridge:**
    Update the `.env.local` (or equivalent environment configuration) in the BubbleLabs studio directory:
    ```bash
    # Redirects frontend requests to the plugin backend
    VITE_API_URL=http://localhost:8002
    ```

2.  **Initialize the Plugin Singleton:**
    The plugin must be initialized in the browser context. This can be done by importing the distribution bundle and calling the `initialize` method:
    ```javascript
    import { pygraphistryPlugin } from '@openevolve/bubblelab-pygraphistry-plugin';

    await pygraphistryPlugin.initialize({
      apiKey: "YOUR_GRAPHISTRY_KEY", // Optional: uses env var if omitted
      gpuAcceleration: true
    });
    ```

---

## 3. Manual Feature Toggling

By design, all advanced visualizations are **disabled by default** to preserve system resources. To begin using them:

1.  Open the **Visualization Settings Panel** (rendered via `VizSettingsPanel.tsx`).
2.  **Toggle the Switch** for the desired modules (e.g., PyGraphistry, Causal Discovery, NeuroMANCER).
3.  The plugin will immediately begin fetching real-time analytical data from the backend for the selected `BubbleFlow`.

---

## Technical Support & Architecture
For a deeper dive into the symmetric adapter pattern and convention bridging used here, refer to the [PLUGIN_ARCHITECTURE.md](../../PLUGIN_ARCHITECTURE.md) in the root directory.
