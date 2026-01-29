# 🏗️ BubbleLabs Plugin Merge - Architecture Diagrams

**Visual representation of plugin architecture and merge strategy.**

---

## 📊 Current Architecture

### Plugin 1: openevolve-bubblelab-plugin

```
┌─────────────────────────────────────────────────────────────┐
│              openevolve-bubblelab-plugin                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Component Layer                         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • OpenEvolveNodeWrapper                             │   │
│  │  • EvolutionConfigPanel                              │   │
│  │  • AdversarialConfigPanel                            │   │
│  │  • DecompositionConfigPanel                          │   │
│  │  • IntegrationConfigPanel                            │   │
│  │  • EnhancedOpenEvolveConfigPanel                     │   │
│  │  • PerformanceConfigTab                              │   │
│  │  • SecurityConfigTab                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Node Layer                           │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • EvolutionNode                                     │   │
│  │  • AdversarialNode                                   │   │
│  │  • DecompositionNode                                 │   │
│  │  • KnowledgeQueryNode                                │   │
│  │  • LeanAIDENode                                      │   │
│  │  • HephaestusNode                                    │   │
│  │  • MDAPNode                                          │   │
│  │  • MAKERNode                                         │   │
│  │  • NodeRegistry                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Hook Layer                            │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • useOpenEvolvePlugin                               │   │
│  │  • useEvolution                                      │   │
│  │  • useAdversarial                                    │   │
│  │  • useDecomposition                                  │   │
│  │  • useKnowledgeEngine                                │   │
│  │  • useLeanAIDE                                       │   │
│  │  • useHephaestus                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Utility Layer                          │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • createOpenEvolvePlugin                            │   │
│  │  • createEnhancedOpenEvolvePlugin                    │   │
│  │  • validateConfig                                    │   │
│  │  • normalizeConfig                                   │   │
│  │  • mergeConfigs                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Type Layer                            │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • plugin-types.ts                                   │   │
│  │  • enhanced-plugin-types.ts                          │   │
│  │  • extended-plugin-types.ts                          │   │
│  │  • nodes.ts                                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Plugin 2: leanaide-bubblelab-plugin

```
┌─────────────────────────────────────────────────────────────┐
│               leanaide-bubblelab-plugin                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Component Layer                         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • LeanAideVerification                              │   │
│  │  • LeanAidePanel                                     │   │
│  │  • RagbitsKnowledgeSearch                            │   │
│  │  • AnalyticsDashboard                                │   │
│  │  • KnowledgeGraphIntegration                         │   │
│  │  • EnhancedLeanAideVerification                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Integration Layer                       │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • LeanAideBubbleLabIntegration                      │   │
│  │  • BubbleLabIntegration                              │   │
│  │  • LeanAideAutoformalizationEngine                   │   │
│  │  • autoformalize_with_mdap_maker                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Plugin System Layer                     │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • LeanAidePlugin                                    │   │
│  │  • PluginManager                                     │   │
│  │  • PluginManagerProvider                             │   │
│  │  • PluginRegistry                                    │   │
│  │  • usePluginManager                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Service Layer                          │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • LeanAideClient                                    │   │
│  │  • RagbitsClient                                     │   │
│  │  • leanaideService                                   │   │
│  │  • ragbitsService                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Hook Layer                            │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • useAutoformalizationAnalytics                     │   │
│  │  • usePluginManager                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Proposed Merged Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│            openevolve-bubblelab-plugin (MERGED)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    UI Layer                                  │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  ┌──────────────────┐  ┌──────────────────┐                │    │
│  │  │  Evolution UI    │  │  Verification UI │                │    │
│  │  │  - Config Panels │  │  - LeanAIDE      │                │    │
│  │  │  - Node Editors  │  │  - RAGBits       │                │    │
│  │  └──────────────────┘  └──────────────────┘                │    │
│  │                                                              │    │
│  │  ┌──────────────────┐  ┌──────────────────┐                │    │
│  │  │ Knowledge UI     │  │  Analytics UI    │                │    │
│  │  │  - Search        │  │  - Dashboard     │                │    │
│  │  │  - Graph         │  │  - Metrics       │                │    │
│  │  └──────────────────┘  └──────────────────┘                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                  ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Node System Layer                         │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  ┌────────────────┐  ┌────────────────┐                    │    │
│  │  │ Evolution Nodes│  │Verification Node│                    │    │
│  │  │ • MCTS         │  │ • LeanAIDE      │                    │    │
│  │  │ • Genetic      │  │ • Autoformal.  │                    │    │
│  │  │ • Hybrid       │  └────────────────┘                    │    │
│  │  └────────────────┘                                        │    │
│  │                                                              │    │
│  │  ┌────────────────┐  ┌────────────────┐                    │    │
│  │  │Adversarial Nodes│  │ Knowledge Nodes │                    │    │
│  │  │ • Red-team     │  │ • Query         │                    │    │
│  │  │ • Attack gen   │  │ • Search        │                    │    │
│  │  └────────────────┘  └────────────────┘                    │    │
│  │                                                              │    │
│  │  ┌────────────────┐  ┌────────────────┐                    │    │
│  │  │Decomposition N. │  │ Integration N.  │                    │    │
│  │  │ • Hierarchical │  │ • MDAP          │                    │    │
│  │  │ • Adaptive     │  │ • MAKER         │                    │    │
│  │  └────────────────┘  └────────────────┘                    │    │
│  │                                                              │    │
│  │           Unified NodeRegistry (Type-Safe)                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                  ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                Service Layer (Unified)                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │    │
│  │  │  Evolution Service  │  │  LeanAIDE Service   │          │    │
│  │  │  • MCTS engine      │  │  • Translation      │          │    │
│  │  │  • Population mgmt  │  │  • Verification     │          │    │
│  │  └─────────────────────┘  └─────────────────────┘          │    │
│  │                                                              │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │    │
│  │  │ Knowledge Service   │  │  RAGBits Service    │          │    │
│  │  │  • Graph queries    │  │  • Semantic search  │          │    │
│  │  │  • Artifact mgmt    │  │  • Ingestion        │          │    │
│  │  └─────────────────────┘  └─────────────────────┘          │    │
│  │                                                              │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │    │
│  │  │  Hephaestus Service │  │ Analytics Service   │          │    │
│  │  │  • Delegation       │  │  • Metrics          │          │    │
│  │  │  • Orchestration    │  │  • Tracking         │          │    │
│  │  └─────────────────────┘  └─────────────────────┘          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                  ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Plugin System Layer                       │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  Unified PluginManager                                       │    │
│  │  ├─ PluginRegistry (Type-discriminated)                      │    │
│  │  ├─ PluginLoader                                             │    │
│  │  ├─ Lifecycle Manager                                        │    │
│  │  └─ Event Bus                                                │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                  ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Core Infrastructure                       │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │    │
│  │  │  Type System    │  │   Config System │                  │    │
│  │  │  • Unified defs │  │  • Schema       │                  │    │
│  │  │  • Namespaced   │  │  • Validation   │                  │    │
│  │  └─────────────────┘  └─────────────────┘                  │    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │    │
│  │  │   Hooks Layer   │  │   Utilities     │                  │    │
│  │  │  • useEvolution │  │  • Validation   │                  │    │
│  │  │  • useLeanAIDE  │  │  • Merging      │                  │    │
│  │  │  • useKnowledge │  │  • Adapters     │                  │    │
│  │  └─────────────────┘  └─────────────────┘                  │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 Merge Flow Diagram

```
┌──────────────────────┐
│  ORIGINAL PLUGINS    │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────┐       ┌────┐
│ P1 │       │ P2 │
│    │       │    │
└────┘       └────┘
 │            │
 │            │
 └────┬───────┘
      │
      ▼
 ┌─────────┐
 │ PHASE 1 │ Discovery
 │ Agent 1 │ Analysis
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ PHASE 2 │ Architecture
 │ Agent 2 │ Design
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ PHASE 3 │ Code
 │ Agent 3 │ Migration
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ PHASE 4 │ Integration
 │ Agent 4 │ Resolution
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ PHASE 5 │ Testing
 │ Agent 5 │ Validation
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ PHASE 6 │ Docs
 │ Agent 6 │ Release
 └────┬────┘
      │
      ▼
 ┌──────────────────┐
 │  MERGED PLUGIN   │
 │  ✓ All features  │
 │  ✓ Zero loss     │
 │  ✓ Tested        │
 └──────────────────┘
```

---

## 📦 Directory Mapping

### Before Merge

```
openevolve-bubblelab-plugin/          leanaide-bubblelab-plugin/
├── src/                              ├── src/
│   ├── components/                   │   ├── components/
│   │   ├── Enhanced...Panel.tsx      │   │   ├── LeanAideVerification.tsx
│   │   ├── OpenEvolve...Panel.tsx    │   │   ├── LeanAidePanel.tsx
│   │   ├── nodes/                    │   │   └── RagbitsKnowledgeSearch.tsx
│   │   └── tabs/                     │   ├── lib/
│   ├── nodes/                        │   │   ├── leanaideClient.ts
│   │   ├── DecompositionNode.ts      │   │   └── ragbitsClient.ts
│   │   ├── EvolutionNode.ts          │   ├── services/
│   │   ├── registry.ts               │   │   ├── leanaideService.ts
│   │   └── ...                       │   │   └── ragbitsService.ts
│   ├── types/                        │   ├── integration/
│   │   ├── plugin-types.ts           │   │   └── autoformalizationAnalytics.tsx
│   │   ├── enhanced-plugin-types.ts  │   ├── plugins/
│   │   └── nodes.ts                  │   │   ├── LeanAidePlugin.tsx
│   ├── hooks/                        │   │   └── PluginRegistry.tsx
│   │   └── useEnhanced...Config.ts   │   └── index.ts
│   ├── utils/                        └── ...
│   │   └── createPlugin.ts
│   └── index.ts
└── ...
```

### After Merge

```
openevolve-bubblelab-plugin-merged/
├── src/
│   ├── core/                          [SHARED INFRASTRUCTURE]
│   │   ├── types/                     [Unified type definitions]
│   │   │   ├── index.ts
│   │   │   ├── evolution.types.ts
│   │   │   ├── verification.types.ts
│   │   │   ├── knowledge.types.ts
│   │   │   └── plugin.types.ts
│   │   ├── constants/                 [Shared constants]
│   │   │   └── index.ts
│   │   └── utils/                     [Shared utilities]
│   │       ├── validation.ts
│   │       ├── merging.ts
│   │       └── adapters.ts
│   │
│   ├── nodes/                         [ALL NODES]
│   │   ├── evolution/
│   │   │   ├── EvolutionNode.ts       [from P1]
│   │   │   ├── MCTSNode.ts            [from P1]
│   │   │   └── index.ts
│   │   ├── adversarial/
│   │   │   ├── AdversarialNode.ts     [from P1]
│   │   │   └── index.ts
│   │   ├── decomposition/
│   │   │   ├── DecompositionNode.ts   [from P1]
│   │   │   └── index.ts
│   │   ├── verification/              [NEW - from P2]
│   │   │   ├── LeanAideVerificationNode.ts
│   │   │   ├── AutoformalizationNode.ts
│   │   │   └── index.ts
│   │   ├── knowledge/
│   │   │   ├── KnowledgeQueryNode.ts  [from P1]
│   │   │   ├── RagbitsSearchNode.ts   [from P2]
│   │   │   └── index.ts
│   │   ├── integration/
│   │   │   ├── MDAPNode.ts            [from P1]
│   │   │   ├── MAKERNode.ts           [from P1]
│   │   │   └── index.ts
│   │   └── registry.ts                [MERGED]
│   │
│   ├── components/                    [ALL UI COMPONENTS]
│   │   ├── nodes/
│   │   │   ├── evolution/             [from P1]
│   │   │   ├── adversarial/           [from P1]
│   │   │   ├── decomposition/         [from P1]
│   │   │   └── verification/          [from P2]
│   │   ├── panels/
│   │   │   ├── EvolutionConfigPanel.tsx        [from P1]
│   │   │   ├── AdversarialConfigPanel.tsx      [from P1]
│   │   │   ├── DecompositionConfigPanel.tsx    [from P1]
│   │   │   ├── LeanAideVerificationPanel.tsx   [from P2]
│   │   │   ├── KnowledgeSearchPanel.tsx        [from P2]
│   │   │   └── UnifiedConfigPanel.tsx          [MERGED]
│   │   ├── tabs/
│   │   │   ├── PerformanceTab.tsx      [from P1]
│   │   │   ├── SecurityTab.tsx         [from P1]
│   │   │   ├── VerificationTab.tsx     [from P2]
│   │   │   └── AnalyticsTab.tsx        [from P2]
│   │   └── index.ts
│   │
│   ├── services/                      [ALL SERVICES]
│   │   ├── evolution/
│   │   │   └── evolutionService.ts    [from P1]
│   │   ├── leanaide/                   [from P2]
│   │   │   ├── LeanAideClient.ts
│   │   │   ├── leanaideService.ts
│   │   │   └── index.ts
│   │   ├── ragbits/                    [from P2]
│   │   │   ├── RagbitsClient.ts
│   │   │   ├── ragbitsService.ts
│   │   │   └── index.ts
│   │   ├── knowledge/
│   │   │   └── knowledgeService.ts    [from P1]
│   │   ├── hephaestus/
│   │   │   └── hephaestusService.ts   [from P1]
│   │   └── index.ts
│   │
│   ├── integration/                   [INTEGRATION LAYER]
│   │   ├── autoformalization/
│   │   │   ├── engine.ts              [from P2]
│   │   │   ├── analytics.tsx          [from P2]
│   │   │   └── index.ts
│   │   ├── knowledge-graph/
│   │   │   └── integration.ts         [from P2]
│   │   └── index.ts
│   │
│   ├── plugins/                       [PLUGIN SYSTEM]
│   │   ├── LeanAidePlugin.ts          [from P2]
│   │   ├── PluginManager.tsx          [MERGED]
│   │   ├── PluginRegistry.ts          [MERGED]
│   │   └── index.ts
│   │
│   ├── hooks/                         [ALL HOOKS]
│   │   ├── evolution.ts               [from P1]
│   │   ├── adversarial.ts             [from P1]
│   │   ├── decomposition.ts           [from P1]
│   │   ├── leanaide.ts                [from P2]
│   │   ├── knowledge.ts               [from P1 + P2]
│   │   ├── pluginManager.tsx          [from P2]
│   │   └── index.ts
│   │
│   ├── utils/                         [PLUGIN UTILITIES]
│   │   ├── createPlugin.ts            [MERGED]
│   │   ├── config.ts                  [MERGED]
│   │   └── index.ts
│   │
│   └── index.ts                       [MAIN EXPORT]
│
├── examples/                          [MERGED EXAMPLES]
├── public/                            [MERGED ASSETS]
├── docs/                              [DOCUMENTATION]
├── tests/                             [ALL TESTS]
├── README.md
├── MIGRATION_GUIDE.md
├── API_REFERENCE.md
└── package.json
```

---

## 🎨 Feature Integration Map

### Feature Matrix

| Feature | Plugin 1 | Plugin 2 | Merged Location | Notes |
|---------|----------|----------|-----------------|-------|
| **Evolution** | ✅ | ❌ | `nodes/evolution/` | Keep as-is |
| **Adversarial** | ✅ | ❌ | `nodes/adversarial/` | Keep as-is |
| **Decomposition** | ✅ | ❌ | `nodes/decomposition/` | Keep as-is |
| **LeanAIDE Verif** | ❌ | ✅ | `nodes/verification/` | New location |
| **Autoformalization** | ❌ | ✅ | `integration/autoformalization/` | Keep as-is |
| **Knowledge Query** | ✅ | ❌ | `nodes/knowledge/` | Keep as-is |
| **RAGBits Search** | ❌ | ✅ | `nodes/knowledge/` | Merge here |
| **MDAP Integration** | ✅ | ❌ | `nodes/integration/` | Keep as-is |
| **MAKER Integration** | ✅ | ❌ | `nodes/integration/` | Keep as-is |
| **Hephaestus** | ✅ | ❌ | `services/hephaestus/` | Keep as-is |
| **LeanAIDE Client** | ❌ | ✅ | `services/leanaide/` | Keep as-is |
| **RAGBits Client** | ❌ | ✅ | `services/ragbits/` | Keep as-is |
| **Analytics** | ❌ | ✅ | `components/analytics/` | New location |
| **Plugin System** | ❌ | ✅ | `plugins/` | Keep as-is |
| **Node Registry** | ✅ | ❌ | `nodes/registry.ts` | Extend for P2 |

---

## 🔄 Data Flow Integration

### Before Merge (Separate)

```
┌──────────────┐         ┌──────────────┐
│   Plugin 1   │         │   Plugin 2   │
│              │         │              │
│  Evolution → │         │  LeanAIDE →  │
│  Adversarial │         │  RAGBits →   │
│  Knowledge   │         │  Analytics   │
│              │         │              │
└──────────────┘         └──────────────┘
       │                         │
       └─────────┬───────────────┘
                 │
                 ▼
          ┌─────────────┐
          │  BubbleLab  │
          └─────────────┘
```

### After Merge (Integrated)

```
┌─────────────────────────────────────────────┐
│         openevolve-bubblelab-plugin         │
│                                             │
│  ┌────────────┐      ┌────────────┐        │
│  │ Evolution  │──────│ Knowledge  │        │
│  │            │      │            │        │
│  └────────────┘      └──────┬─────┘        │
│       │                   │                │
│       │            ┌──────▼─────┐          │
│       └───────────▶│ LeanAIDE   │◀─────────┤
│                    │            │          │
│                    └──────┬─────┘          │
│                           │                │
│                    ┌──────▼─────┐          │
│                    │ RAGBits    │          │
│                    │            │          │
│                    └──────┬─────┘          │
│                           │                │
│                    ┌──────▼─────┐          │
│                    │ Analytics  │          │
│                    │            │          │
│                    └──────┬─────┘          │
└──────────────────────────┼─────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  BubbleLab  │
                    └─────────────┘
```

---

## 📊 Integration Points

### 1. Evolution + LeanAIDE
```typescript
// Evolution nodes can use LeanAIDE for verification
EvolutionNode {
  verification: {
    service: 'leanaide',
    autoVerify: true
  }
}
```

### 2. Knowledge + RAGBits
```typescript
// Knowledge nodes use RAGBits for semantic search
KnowledgeQueryNode {
  search: {
    backend: 'ragbits',
    semantic: true
  }
}
```

### 3. Decomposition + Analytics
```typescript
// Decomposition tracks metrics in analytics
DecompositionNode {
  analytics: {
    enabled: true,
    trackDecompositionTime: true
  }
}
```

### 4. All + Plugin System
```typescript
// All features accessible through unified plugin system
PluginManager {
  plugins: {
    evolution: EvolutionPlugin,
    leanaide: LeanAidePlugin,
    knowledge: KnowledgePlugin
  }
}
```

---

## ✅ Validation Checklist

### Structural Validation
- [ ] All files from P1 present in merge
- [ ] All files from P2 present in merge
- [ ] Directory structure matches plan
- [ ] Naming conventions consistent
- [ ] No duplicate files

### Functional Validation
- [ ] All P1 features working
- [ ] All P2 features working
- [ ] Integrations functional
- [ ] No breaking changes (without adapter)
- [ ] Backward compatibility maintained

### Type Validation
- [ ] All TypeScript types defined
- [ ] No `any` types without justification
- [ ] Exports properly typed
- [ ] Imports resolve correctly
- [ ] No circular dependencies

### Build Validation
- [ ] `npm install` successful
- [ ] `npm run build` successful
- [ ] No build warnings
- [ ] Bundle size acceptable
- [ ] Tree-shaking working

---

**END OF ARCHITECTURE DOCUMENTATION**
