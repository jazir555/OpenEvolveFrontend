/**
 * BubbleLab Components Manifest
 *
 * Typed view over `src/component-manifest.json`, the single source of truth for
 * every configuration knob exposed by the `@openevolve/bubblelab-components`
 * package. The Python BubbleLab UI bridge reads the same JSON file so each knob
 * is reachable from either side of the integration.
 *
 * This module is part of the package `tsc` build (it lives under `src/lib`,
 * which `tsconfig.json` includes) so the manifest is compiled into `dist` and
 * ships with the published package.
 */

import manifest from "../component-manifest.json";

export type ControlKind =
  | "select"
  | "slider"
  | "number"
  | "boolean"
  | "text"
  | "textarea"
  | "json"
  | "multiselect"
  | "password"
  | "button"
  | "display";

export interface ConfigKnob {
  id: string;
  label: string;
  control: ControlKind;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  default?: unknown;
}

export type ComponentCategory = "sidebar" | "main" | "bubblelabs";

export interface ComponentManifestEntry {
  id: string;
  name: string;
  category: ComponentCategory;
  source: string;
  knobs: ConfigKnob[];
}

export interface BubbleLabComponentManifest {
  package: string;
  version: string;
  generatedFrom: string;
  description: string;
  components: ComponentManifestEntry[];
}

/** Parsed constant manifest. Safe to import in both Node and browser builds. */
export const bubbleLabComponentManifest = manifest as BubbleLabComponentManifest;

/**
 * Return every configuration knob across all components, de-duplicated by id
 * (later definitions win). This is the "every config knob is accessible" surface
 * the Python bridge consumes.
 */
export function getAllConfigKnobs(): ConfigKnob[] {
  const byId = new Map<string, ConfigKnob>();
  for (const component of bubbleLabComponentManifest.components) {
    for (const knob of component.knobs) {
      byId.set(knob.id, knob);
    }
  }
  return [...byId.values()];
}

/** Return the full manifest object (components, package metadata, etc.). */
export function getComponentManifest(): BubbleLabComponentManifest {
  return bubbleLabComponentManifest;
}

/** Look up a single knob by id across all components. */
export function getConfigKnob(id: string): ConfigKnob | undefined {
  return getAllConfigKnobs().find((knob) => knob.id === id);
}

/** Return the knob ids exposed by one component. */
export function getComponentKnobIds(componentId: string): string[] {
  const component = bubbleLabComponentManifest.components.find((c) => c.id === componentId);
  return component ? component.knobs.map((k) => k.id) : [];
}
