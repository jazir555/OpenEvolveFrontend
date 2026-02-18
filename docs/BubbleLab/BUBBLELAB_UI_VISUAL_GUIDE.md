# 🎨 BubbleLab + OpenEvolve UI Visual Guide

**What it actually looks like when you use it**

---

## 📱 THE COMPLETE UI

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              BubbleLab Studio                                           ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  ┌─────────┐ ┌───────────────────────────────────────────────────────────────────┐ ║
║  │ Sidebar │ │                         Main Content                          │ ║
║  │         │ │                                                                   │ ║
║  │  🏠     │ │                                                                   │ ║
║  │  💬     │ │                                                                   │ ║
║  │  ⚡     │ │                                                                   │ ║
║  │  ─────  │ │                                                                   │ ║
║  │         │ │                                                                   │ ║
║  │ 🧬 OpenE│ │                    OpenEvolve Dashboard                            │ ║
║  │  volve  │ │                                                                   │ ║
║  │   ├Dash│ │  ┌─────────────────────┬───────────────────────┐                │ ║
║  │   ├Anal│ │  │                     │                       │                │ ║
║  │   ├Know│ │  │   Config Panel       │   Execution Monitor    │                │ ║
║  │   └Lean│ │  │                     │                       │                │ ║
║  │         │ │  ┌───────────────────┐ │  ┌─────────────────────┐│                │ ║
║  │  📊     │ │  │ Service: Evolution │ │  │ Progress: 70%        ││                │ ║
║  │  ⚙️     │ │  └───────────────────┘ │  │ ● Connected         ││                │ ║
║  │         │ │                       │  │                      ││                │ ║
║  │         │ │  Content:              │  │ Live Logs:           ││                │ ║
║  │         │ │  ┌─────────────────┐  │  │ [10:30] Starting...  ││                │ ║
║  │         │ │  │ Optimize this   │  │  │ [10:31] Gen 1/10     ││                │ ║
║  │         │ │  │ content...      │  │  │ [10:32] Gen 2/10     ││                │ ║
║  │         │ │  └─────────────────┘  │  │ [10:33] Gen 3/10     ││                │ ║
║  │         │ │                       │  │ ...                  ││                │ ║
║  │         │ │  Iterations:    [10]  │  │                      ││                │ ║
║  │         │ │  Temperature:   [0.7] │  │                      ││                │ ║
║  │         │ │  Population:    [5]   │  │                      ││                │ ║
║  │         │ │                       │  │                      ││                │ ║
║  │         │ │  Provider: [Anthropic▼]│  │                      ││                │ ║
║  │         │ │  Model:    [Claude 3▼] │  │                      ││                │ ║
║  │         │ │                       │  │                      ││                │ ║
║  │         │ │  [Start] [Stop]        │  │ [View Results]       ││                │ ║
║  │         │ │                       │  └─────────────────────┘│                │ ║
║  │         │ │  └─────────────────────┘                       └─────────────────────┘│                │ ║
║  │         │ │                                                                   │ ║
║  └─────────┘ └───────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🔄 CREATING A WORKFLOW - STEP BY STEP

### Step 1: Open Bubble Builder

```
Click "💬 Bubbles" in sidebar
→ Click "New Bubble" button
```

### Step 2: Choose OpenEvolve Service

```
┌─────────────────────────────────────┐
│  Add Service to Bubble              │
├─────────────────────────────────────┤
│  🔍 Search services...             │
│                                     │
│  OpenEvolve Services:               │
│  ┌─────────────────────────────┐   │
│  │ 🧬 Evolution Engine         │   │
│  │    Genetic algorithm        │   │
│  │    + Add                     │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │ ⚔️ Adversarial Testing       │   │
│  │    Red team vs blue team    │   │
│  │    + Add                     │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │ 🎯 MDAP                     │   │
│  │    Multi-domain planning    │   │
│  │    + Add                     │   │
│  └─────────────────────────────┘   │
│                                     │
│  Or search other services...         │
└─────────────────────────────────────┘
```

### Step 3: Configure the Service

```
Click "Evolution Engine"
→ Panel slides in from the right →

┌─────────────────────────────────────┐
│  Evolution Configuration            │
├─────────────────────────────────────┤
│                                     │
│  Content to Evolve                  │
│  ┌─────────────────────────────┐   │
│  │ Write better documentation  │   │
│  │                             │   │
│  └─────────────────────────────┘   │
│                                     │
│  Number of Iterations               │
│  ┌─────────────────────────────┐   │
│  0    5    10   15   20         │   │
│  └─────────────────────────────┘   │
│                                     │
│  Temperature (Creativity)           │
│  ┌─────────────────────────────┐   │
│  0.0──────0.7──────1.0──────2.0  │   │
│  └─────────────────────────────┘   │
│                                     │
│  AI Provider                        │
│  ┌─────────────────────────────┐   │
│  [Anthropic          ▼]         │   │
│  ├─ OpenAI                       │   │
│  ├─ Anthropic                    │   │
│  └─ Google                       │   │
│  └─────────────────────────────┘   │
│                                     │
│  Model                              │
│  ┌─────────────────────────────┐   │
│  [Claude 3 Sonnet     ▼]         │   │
│  ├─ GPT-4                        │   │
│  ├─ Claude 3 Sonnet               │   │
│  └─ Claude 3 Opus                 │   │
│  └─────────────────────────────┘   │
│                                     │
│  [Cancel]              [Apply]     │
└─────────────────────────────────────┘
```

### Step 4: Connect to Other Services

```
Drag from Evolution output → Another service input

Evolution ──────> LeanAide ──────> Knowledge Base
   ↓                    ↓                 ↓
Store result       Verify result    Save as artifact
```

### Step 5: Execute Workflow

```
Click "▶ Execute Workflow" button (top right)
→ Execution panel slides in →

┌─────────────────────────────────────┐
│  Executing: Content Evolution       │
├─────────────────────────────────────┤
│                                     │
│  Overall Progress                    │
│  ████████████░░░░░░░░░░ 70%        │
│                                     │
│  Current Step: Generation 7/10      │
│                                     │
│  Live Logs (auto-scrolling)         │
│  [10:30:00] Starting evolution...   │
│  [10:30:05] Generation 1 complete   │
│  [10:30:15] Generation 2 complete   │
│  [10:30:25] Generation 3 complete   │
│  [10:30:35] Generation 4 complete   │
│  [10:30:45] Generation 5 complete   │
│  [10:30:55] Generation 6 complete   │
│  [10:31:05] Generation 7 complete   │
│  ...                                │
│                                     │
│  Est. time remaining: 2 min         │
│                                     │
│  [Pause] [Stop] [View Results]      │
└─────────────────────────────────────┘
```

---

## 📊 ANALYTICS DASHBOARD

```
URL: http://localhost:3000/openevolve/analytics

┌─────────────────────────────────────────────────────────────────┐
│  Analytics Dashboard                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Date Range: [2025-01-01] to [2025-01-06]     [Last 30 days ▼]  │
│                                                                 │
│  ┌────────────┬────────────┬────────────┬────────────┐            │
│  │ Total      │ Success    │ Avg Time   │ Best       │            │
│  │ Evolutions │ Rate       │ per Evo    │ Result     │            │
│  ├────────────┼────────────┼────────────┼────────────┤            │
│  │   1,247    │   94.5%    │   2.3 min  │   98.7%    │            │
│  └────────────┴────────────┴────────────┴────────────┘            │
│                                                                 │
│  Performance Over Time                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 100│                                    ╱╲               │    │
│  │  90│                        ╱╲       ╱  ╲    ╱  ╲         │    │
│  │  80│                   ╱  ╲    ╱    ╲  ╱    ╲       │    │
│  │  70│              ╱    ╲  ╱      ╲╱      ╲      │    │
│  │  60│         ╱     ╲  ╱        ╲        │       │    │
│  │  50│    ╱    ╲    ╱  ╱          ╲       │       │    │
│  │  40│   ╱      ╲  ╱   │           ╲      │       │    │
│  │  30│ ╱        ╲╱    │            │     │       │    │
│  │   └───────────────────────────────────────────     │    │
│  │     Mon   Tue   Wed   Thu   Fri   Sat   Sun      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                                 │
│  Recent Results                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ID     │ Service     │ Status   │ Score   │ Date        │    │
│  ├────────┼─────────────┼─────────┼─────────┼────────────┤    │
│  │ #1234  │ Evolution   │ ✅      │ 98.7%   │ 2 min ago   │    │
│  │ #1233  │ Adversarial │ ✅      │ Passed  │ 5 min ago   │    │
│  │ #1232  │ Evolution   │ ✅      │ 97.2%   │ 8 min ago   │    │
│  │ #1231  │ Maker       │ ⚠️      │ 85.3%   │ 12 min ago  │    │
│  └────────┴─────────────┴─────────┴─────────┴────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 KNOWLEDGE BASE

```
URL: http://localhost:3000/openevolve/knowledge

┌─────────────────────────────────────────────────────────────────┐
│  Knowledge Base                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔍 Search: [Enter search terms...]                    [Search] │
│                                                                 │
│  Filters: [All Types ▼] [All Tags ▼] [All Dates ▼]             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ┌─────────────┬──────────────────────────────────────┐ │    │
│  │  │ Thumbnail    │ Title                              │ │    │
│  │  │             │                                   │ │    │
│  │  │  📄         │ Evolution Strategy Doc            │ │    │
│  │  │             │ Tags: strategy, evolution           │ │    │
│  │  │             │ Created: Jan 5, 2025               │ │    │
│  │  ├─────────────┼──────────────────────────────────────┤ │    │
│  │  │  🤖         │ Adversarial Test Results           │ │    │
│  │  │             │ Tags: testing, security             │ │    │
│  │  │             │ Created: Jan 4, 2025               │ │    │
│  │  ├─────────────┼──────────────────────────────────────┤ │    │
│  │  │  📊         │ Performance Metrics Q4            │ │    │
│  │  │             │ Tags: analytics, metrics            │ │    │
│  │  │             │ Created: Jan 3, 2025               │ │    │
│  │  └─────────────┴──────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  [+ New Artifact]                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 LEANAIDE INTERFACE

```
URL: http://localhost:3000/openevolve/leanaide

┌─────────────────────────────────────────────────────────────────┐
│  LeanAide - Lean 4 Proof Assistant                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model: [Claude 3 Opus ▼]                                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Lean Code Editor (Monaco)                              │    │
│  │                                                           │    │
│  │ theorem proof_example :                                │    │
│  │   n : Nat                                               │    │
│  │   h : n → result                                      │    │
│  │   by                                                   │    │
│  │   induction n.m                                      │    │
│  │   case n                                              │    │
│  │   zero                                               │    │
│  │   simp                                               │    │
│  │     h (n.succ)                                    │    │
│  │                                                   │    │
│  │  [Verify]  [Explain]  [Get Help]                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Verification Status                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ✅ Proof Verified                                       │    │
│  │                                                           │    │
│  │ The proof is correct. All steps type-check successfully. │    │
│  │                                                           │    │
│  │ Steps: 5 │ Tactics: 2 │ Time: 0.3s                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  [Save to Knowledge Base]  [Export as Lean File]              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ CONFIGURATION PANELS

### Evolution Engine Config

```
┌─────────────────────────────────────┐
│  ⚙️ Evolution Engine Configuration  │
├─────────────────────────────────────┤
│                                     │
│  📝 Content to Evolve               │
│  ┌───────────────────────────────┐  │
│  │ Enter your content here...    │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  🔄 Number of Iterations            │
│     [1════════════10]  10          │
│                                     │
│  🌡️ Temperature                     │
│     [0.0──────0.7──────1.0]       │
│     Conservative ──── Creative       │
│                                     │
│  👥 Population Size                 │
│     [1════════════20]  5           │
│                                     │
│  🤖 AI Provider                     │
│     ┌─────────────────────────┐   │
│     │ Anthropic              │   │
│     ├─ OpenAI                │   │
│     ├─ Anthropic              │   │
│     └─ Google                │   │
│     └─────────────────────────┘   │
│                                     │
│  🧠 Model                           │
│     ┌─────────────────────────┐   │
│     │ Claude 3 Sonnet        │   │
│     ├─ GPT-4                 │   │
│     ├─ Claude 3 Sonnet        │   │
│     └─ Claude 3 Opus          │   │
│     └─────────────────────────┘   │
│                                     │
│  📊 Advanced Settings                │
│     ├─ Mutation Rate: [0.1]       │
│     ├─ Crossover Rate: [0.5]       │
│     └─ Elitism: [True]            │
│                                     │
│  [Reset Defaults]  [Apply]  [Cancel]│
└─────────────────────────────────────┘
```

### Adversarial Testing Config

```
┌─────────────────────────────────────┐
│  ⚔️ Adversarial Testing Config        │
├─────────────────────────────────────┤
│                                     │
│  🎯 Target Content                  │
│  ┌───────────────────────────────┐  │
│  │ System prompt or content...   │  │
│  └───────────────────────────────┘  │
│                                     │
│  ⚔️ Attack Mode                     │
│     ┌─────────────────────────┐   │
│     │ Prompt Injection         │   │
│     ├─ Jailbreak               │   │
│     ├─ Adversarial Example     │   │
│     └─ Model Extraction        │   │
│     └─────────────────────────┘   │
│                                     │
│  🔴 Red Team Provider              │
│     ┌─────────────────────────┐   │
│     │ OpenAI (GPT-4)          │   │
│     └─────────────────────────┘   │
│                                     │
│  🔵 Blue Team Provider            │
│     ┌─────────────────────────┐   │
│     │ Anthropic (Claude 3)     │   │
│     └─────────────────────────┘   │
│                                     │
│  ⚔️ Battle Rounds                   │
│     [1════════════10]  3           │
│                                     │
│  🛡️ Defense Strategy                │
│     ┌─────────────────────────┐   │
│     │ Strict Filtering         │   │
│     ├─ Adversarial Training    │   │
│     └─ Prompt Engineering     │   │
│     └─────────────────────────┘   │
│                                     │
│  [Start Battle]  [Stop]            │
└─────────────────────────────────────┘
```

---

## 🎯 QUICK REFERENCE

### Navigation URLs

| Page | URL |
|------|-----|
| Dashboard | `/openevolve` |
| Analytics | `/openevolve/analytics` |
| Knowledge | `/openevolve/knowledge` |
| LeanAide | `/openevolve/leanaide` |

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open Command Palette | `Cmd/Ctrl + K` |
| Navigate to OpenEvolve | `Cmd/Ctrl + Shift + O` |
| New Workflow | `Cmd/Ctrl + N` |
| Execute Workflow | `Cmd/Ctrl + Enter` |
| Stop Execution | `Cmd/Ctrl + .` |

### Icon Meanings

| Icon | Service |
|------|---------|
| 🧬 | Evolution Engine |
| ⚔️ | Adversarial Testing |
| 🔨 | Maker Engine |
| 🎯 | MDAP |
| 🧩 | Decomposition |
| 📚 | Knowledge Engine |
| 🤖 | LeanAide |
| 🔧 | crewai |
| 🏛️ | ROMA |
| 💡 | Invention Planner |

---

## 🎉 That's It!

You now have a **complete, visual understanding** of how OpenEvolve integrates into BubbleLab!

**Start using it**:
```bash
cd BubbleLab/apps/bubble-studio
npm install file:../../../OpenEvolve-Plugin
npm run dev
open http://localhost:3000/openevolve
```

**Happy evolving!** 🚀
