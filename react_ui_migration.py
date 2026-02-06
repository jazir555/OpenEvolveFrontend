"""
React UI Migration Helper - License: Apache 2.0

Utilities for migrating UI UI to React.
Generates React components, hooks, and API clients.

Usage:
    python react_ui_migration.py --generate-components
    python react_ui_migration.py --generate-hooks
    python react_ui_migration.py --full-migration
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse

@dataclass
class ComponentSpec:
    """Specification for a React component."""
    name: str
    props: List[Dict]
    state: List[str]
    api_endpoints: List[str]
    description: str


class ReactUIGenerator:
    """
    Generates React UI components from OpenEvolve integration.
    
    Creates:
    - React components for all UI elements
    - Custom hooks for API integration
    - TypeScript types
    - API client
    """
    
    def __init__(self, output_dir: str = "react-ui"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Component specifications
        self.components = [
            ComponentSpec(
                name="WorkflowDashboard",
                props=[
                    {"name": "workflows", "type": "Workflow[]"},
                    {"name": "onWorkflowSelect", "type": "(id: string) => void"}
                ],
                state=["selectedWorkflow", "loading"],
                api_endpoints=["/api/v1/workflows"],
                description="Main workflow management dashboard"
            ),
            ComponentSpec(
                name="ServiceMonitor",
                props=[
                    {"name": "services", "type": "ServiceStatus[]"}
                ],
                state=["refreshing", "lastUpdate"],
                api_endpoints=["/api/v1/services", "/health"],
                description="Real-time service monitoring"
            ),
            ComponentSpec(
                name="KnowledgeExplorer",
                props=[
                    {"name": "patterns", "type": "Pattern[]"},
                    {"name": "artifacts", "type": "Artifact[]"}
                ],
                state=["activeTab", "searchQuery"],
                api_endpoints=["/api/v1/knowledge/patterns", "/api/v1/knowledge/artifacts"],
                description="Knowledge base explorer"
            ),
            ComponentSpec(
                name="DecompositionViewer",
                props=[
                    {"name": "decomposition", "type": "Decomposition"}
                ],
                state=["expandedNodes", "selectedNode"],
                api_endpoints=["/api/v1/decompose"],
                description="Visual decomposition tree viewer"
            ),
            ComponentSpec(
                name="EventStream",
                props=[
                    {"name": "events", "type": "WorkflowEvent[]"}
                ],
                state=["filter", "autoScroll"],
                api_endpoints=["ws://localhost:8080/events"],
                description="Real-time event stream viewer"
            ),
            ComponentSpec(
                name="IntegrationStatus",
                props=[],
                state=["status", "components"],
                api_endpoints=["/api/v1/status"],
                description="System integration status panel"
            ),
        ]
    
    def generate_typescript_types(self) -> str:
        """Generate TypeScript type definitions."""
        types = """// Auto-generated TypeScript types for OpenEvolve React UI
// License: Apache 2.0

export interface Workflow {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  problemDescription: string;
  createdAt: string;
  updatedAt: string;
  stages: WorkflowStage[];
  result?: WorkflowResult;
}

export interface WorkflowStage {
  name: string;
  status: string;
  parameters: Record<string, any>;
  result?: any;
  startedAt?: string;
  completedAt?: string;
}

export interface WorkflowResult {
  fitness?: number;
  solution?: any;
  metrics?: Record<string, number>;
}

export interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  port?: number;
  uptime: number;
  version: string;
}

export interface Pattern {
  id: string;
  type: 'sequence' | 'semantic' | 'parametric' | 'structural';
  description: string;
  confidence: number;
  occurrences: number;
}

export interface Artifact {
  id: string;
  name: string;
  type: 'strategy' | 'template' | 'constraint' | 'heuristic';
  validityScore: number;
  content: any;
}

export interface Decomposition {
  id: string;
  subproblems: SubProblem[];
  entanglementMatrix: Record<string, string[]>;
}

export interface SubProblem {
  id: string;
  description: string;
  complexity: number;
}

export interface WorkflowEvent {
  id: string;
  type: string;
  payload: any;
  timestamp: string;
  priority: number;
}

export interface IntegrationStatus {
  level: string;
  components: ComponentStatus[];
  workflows: number;
  timestamp: string;
}

export interface ComponentStatus {
  name: string;
  status: string;
  version: string;
  capabilities: string[];
}
"""
        return types
    
    def generate_api_client(self) -> str:
        """Generate API client."""
        client = '''// Auto-generated API client for OpenEvolve React UI
// License: Apache 2.0

import {
  Workflow, ServiceStatus, Pattern, Artifact,
  Decomposition, WorkflowEvent, IntegrationStatus
} from './types';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const GRAPHQL_URL = process.env.REACT_APP_GRAPHQL_URL || 'http://localhost:8001/graphql';

class OpenEvolveAPI {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  // Workflows
  async getWorkflows(): Promise<Workflow[]> {
    const data = await this.request<{ workflows: Workflow[] }>('/api/v1/workflows');
    return data.workflows;
  }
  
  async getWorkflow(id: string): Promise<Workflow> {
    return this.request<Workflow>(`/api/v1/workflows/${id}`);
  }
  
  async createWorkflow(problem: string, strategy: string): Promise<Workflow> {
    return this.request<Workflow>('/api/v1/workflows', {
      method: 'POST',
      body: JSON.stringify({ problemDescription: problem, strategy }),
    });
  }
  
  // Services
  async getServices(): Promise<ServiceStatus[]> {
    const data = await this.request<{ services: ServiceStatus[] }>('/api/v1/services');
    return data.services;
  }
  
  async getHealth(): Promise<any> {
    return this.request('/health');
  }
  
  // Knowledge
  async getPatterns(): Promise<Pattern[]> {
    const data = await this.request<{ patterns: Pattern[] }>('/api/v1/knowledge/patterns');
    return data.patterns;
  }
  
  async getArtifacts(): Promise<Artifact[]> {
    const data = await this.request<{ artifacts: Artifact[] }>('/api/v1/knowledge/artifacts');
    return data.artifacts;
  }
  
  // Decomposition
  async decomposeProblem(problem: string, strategy: string): Promise<Decomposition> {
    return this.request<Decomposition>('/api/v1/decompose', {
      method: 'POST',
      body: JSON.stringify({ problemDescription: problem, strategy }),
    });
  }
  
  // GraphQL
  async graphqlQuery(query: string, variables?: any): Promise<any> {
    const response = await fetch(GRAPHQL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, variables }),
    });
    
    return response.json();
  }
  
  // WebSocket for events
  connectEventStream(callback: (event: WorkflowEvent) => void): WebSocket {
    const ws = new WebSocket(`ws://localhost:8080/events`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      callback(data);
    };
    
    return ws;
  }
}

export const api = new OpenEvolveAPI();
export default api;
'''
        return client
    
    def generate_hooks(self) -> str:
        """Generate React hooks."""
        hooks = '''// Auto-generated React hooks for OpenEvolve UI
// License: Apache 2.0

import { useState, useEffect, useCallback } from 'react';
import { api } from './api';
import {
  Workflow, ServiceStatus, Pattern, Artifact,
  WorkflowEvent, IntegrationStatus
} from './types';

// useWorkflows hook
export function useWorkflows() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  const fetchWorkflows = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getWorkflows();
      setWorkflows(data);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, []);
  
  useEffect(() => {
    fetchWorkflows();
  }, [fetchWorkflows]);
  
  return { workflows, loading, error, refetch: fetchWorkflows };
}

// useServices hook
export function useServices(pollInterval = 5000) {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchServices = async () => {
      try {
        const data = await api.getServices();
        setServices(data);
      } catch (err) {
        console.error('Failed to fetch services:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchServices();
    const interval = setInterval(fetchServices, pollInterval);
    
    return () => clearInterval(interval);
  }, [pollInterval]);
  
  return { services, loading };
}

// useKnowledge hook
export function useKnowledge() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchKnowledge = async () => {
      try {
        setLoading(true);
        const [patternsData, artifactsData] = await Promise.all([
          api.getPatterns(),
          api.getArtifacts(),
        ]);
        setPatterns(patternsData);
        setArtifacts(artifactsData);
      } catch (err) {
        console.error('Failed to fetch knowledge:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchKnowledge();
  }, []);
  
  return { patterns, artifacts, loading };
}

// useEventStream hook
export function useEventStream() {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const ws = api.connectEventStream((event) => {
      setEvents((prev) => [...prev.slice(-99), event]);
    });
    
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    
    return () => ws.close();
  }, []);
  
  return { events, connected };
}

// useDecomposition hook
export function useDecomposition() {
  const [decomposing, setDecomposing] = useState(false);
  
  const decompose = useCallback(async (problem: string, strategy: string) => {
    try {
      setDecomposing(true);
      return await api.decomposeProblem(problem, strategy);
    } finally {
      setDecomposing(false);
    }
  }, []);
  
  return { decompose, decomposing };
}
'''
        return hooks
    
    def generate_component(self, spec: ComponentSpec) -> str:
        """Generate a React component."""
        props_str = ', '.join([f"{p['name']}: {p['type']}" for p in spec.props])
        if props_str:
            props_str = f"{{ {props_str} }}"
        else:
            props_str = ""
        
        component = f'''// Auto-generated React component: {spec.name}
// License: Apache 2.0

import React, {{ useState, useEffect }} from 'react';
import './{spec.name}.css';

interface {spec.name}Props {{
  {chr(10).join([f"  {p['name']}: {p['type']};" for p in spec.props])}
}}

/**
 * {spec.description}
 * 
 * API Endpoints:
{chr(10).join([f" * - {ep}" for ep in spec.api_endpoints])}
 */
export const {spec.name}: React.FC<{spec.name}Props> = ({props_str}) => {{
  // State
  {chr(10).join([f"  const [{s}, set{s.capitalize()}] = useState<any>(null);" for s in spec.state])}
  
  useEffect(() => {{
    // Component mount logic
    console.log('{spec.name} mounted');
    
    return () => {{
      // Cleanup
      console.log('{spec.name} unmounted');
    }};
  }}, []);
  
  return (
    <div className="{spec.name.lower()}">
      <h2>{spec.name}</h2>
      <p>{spec.description}</p>
      
      {{/* Component content */}}}}
      <div className="content">
        Implementation placeholder
      </div>
    </div>
  );
}};

export default {spec.name};
'''
        return component
    
    def generate_package_json(self) -> str:
        """Generate package.json."""
        package = {
            "name": "openevolve-react-ui",
            "version": "1.0.0",
            "description": "React UI for OpenEvolve Integration System",
            "license": "Apache-2.0",
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.20.0",
                "axios": "^1.6.0",
                "recharts": "^2.10.0",
                "react-query": "^3.39.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "typescript": "^5.3.0",
                "vite": "^5.0.0"
            },
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview"
            }
        }
        return json.dumps(package, indent=2)
    
    def generate_all(self):
        """Generate all React UI files."""
        # Create directories
        src_dir = self.output_dir / "src"
        src_dir.mkdir(exist_ok=True)
        
        components_dir = src_dir / "components"
        components_dir.mkdir(exist_ok=True)
        
        # Generate files
        (src_dir / "types.ts").write_text(self.generate_typescript_types())
        (src_dir / "api.ts").write_text(self.generate_api_client())
        (src_dir / "hooks.ts").write_text(self.generate_hooks())
        
        # Generate components
        for spec in self.components:
            component_file = components_dir / f"{spec.name}.tsx"
            component_file.write_text(self.generate_component(spec))
            
            # Generate CSS
            css_file = components_dir / f"{spec.name}.css"
            css_file.write_text(f"/* Styles for {spec.name} */\n")
        
        # Generate package.json
        (self.output_dir / "package.json").write_text(self.generate_package_json())
        
        print(f"[OK] Generated React UI in {self.output_dir}/")
        print(f"   - {len(self.components)} components")
        print(f"   - TypeScript types")
        print(f"   - API client")
        print(f"   - React hooks")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="React UI Migration Helper")
    parser.add_argument("--output", default="react-ui", help="Output directory")
    parser.add_argument("--full-migration", action="store_true", help="Generate complete UI")
    
    args = parser.parse_args()
    
    generator = ReactUIGenerator(output_dir=args.output)
    generator.generate_all()
    
    print("\n📋 Next Steps:")
    print("   1. cd react-ui")
    print("   2. npm install")
    print("   3. npm run dev")


if __name__ == "__main__":
    main()

