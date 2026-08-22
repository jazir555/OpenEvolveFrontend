# Flexible LLM Team Assignment System

**Complete Guide to Arbitrary LLM/vLLM Assignment**

**Date**: 2026-01-27
**Status**: ✅ **IMPLEMENTED**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Credential Management](#credential-management)
4. [Team Composition](#team-composition)
5. [API Usage](#api-usage)
6. [Frontend Integration](#frontend-integration)
7. [vLLM Handling](#vllm-handling)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

---

## Overview

This system provides **maximum flexibility** for LLM team composition:

### Key Features

✅ **Arbitrary Assignment**: Any LLM can be assigned to any team role
✅ **Unified Credentials**: Single source of truth for API keys
✅ **vLLM Support**: Vision/multimodal models clearly distinguished
✅ **Credential Verification**: Test credentials before using them
✅ **Provider Flexibility**: Support for 10+ LLM providers
✅ **Team Templates**: Pre-configured teams for common workflows

### Supported Providers

| Provider | Vision Support | Key Models |
|----------|---------------|------------|
| OpenAI | ✅ | GPT-4, GPT-4 Turbo, GPT-4 Vision |
| Anthropic | ✅ | Claude 3 Opus, Sonnet, Haiku |
| Google | ✅ | Gemini Pro, Gemini Ultra |
| OpenRouter | ❌ | 100+ models via aggregator |
| Groq | ❌ | Llama 3 70B, 8B |
| DeepSeek | ❌ | DeepSeek Coder |
| OpenAI-Like | ✅ | vLLM, Ollama, custom APIs |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                          │
│  - Team Builder with LLM selection                        │
│  - Credential management tab                              │
│  - vLLM visual separation                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Team Assignment API                             │
│  - Flexible team composition                               │
│  - Credential verification                                  │
│  - vLLM detection and routing                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Unified Credential Manager                         │
│  1. OpenEvolve config (.env files)                         │
│  2. BubbleLab credentials API (saved credentials)            │
│  3. User-provided (runtime input)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Provider Layer                          │
│  OpenAI, Anthropic, Google, Groq, vLLM, etc.              │
└─────────────────────────────────────────────────────────────┘
```

---

## Credential Management

### Three Sources of Truth

#### 1. OpenEvolve Configuration (`.env` file)

**Location**: `BubbleLab/services/openevolve-api/.env`

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=AI...
GOOGLE_PROJECT_ID=my-project

# Others
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...

# Custom/OpenAI-compatible (vLLM, Ollama, etc.)
CUSTOM_LLMS=[{
  "name": "local-vllm",
  "api_key": "dummy",
  "api_base": "http://localhost:8000/v1",
  "models": ["llava-next", "mistral-code"]
}]
```

**When to use**: Development, personal projects, quick setup

#### 2. BubbleLab Credentials Tab

**Location**: BubbleLab Studio → Credentials

**Features**:
- Persistent storage
- Encrypted API keys
- Team sharing
- Usage tracking

**When to use**: Production, team collaboration, shared credentials

#### 3. User-Provided (Runtime)

**Features**:
- Test before saving
- Temporary usage
- One-time verification

**When to use**: Testing new models, quick experiments

### Credential Verification

All credentials are verified before use:

```bash
POST /api/teams/credentials/verify
{
  "provider": "openai",
  "api_key": "sk-...",
  "model_to_test": "gpt-3.5-turbo"
}

Response:
{
  "verified": true,
  "message": "Credential verified successfully",
  "latency_ms": 342,
  "available_models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
  "credential_id": "verified_openai_1234567890"
}
```

---

## Team Composition

### Team Roles

| Role | Purpose | Can Use vLLM? | Typical Models |
|------|---------|---------------|---------------|
| **Blue Team** | Generate solutions | ✅ Yes | Claude, GPT-4, DeepSeek Coder |
| **Red Team** | Attack/evaluate | ✅ Yes | GPT-4, Claude, Gemini |
| **Judge** | Final decisions | ✅ Yes | Claude Sonnet, GPT-4 |
| **Observer** | Watch and learn | Optional | Any model |
| **Arbiter** | Resolve disputes | Optional | High-reasoning models |

### Team Composition Rules

#### Vision Requirement

```json
{
  "require_vision_for_design": true
}
```

If enabled, team must have at least one vLLM member. Automatically enforced.

#### Provider Diversity

```json
{
  "require_diverse_providers": true
}
```

If enabled, team must have LLMs from different providers. Prevents single-point failure.

### Example Team Configurations

#### Standard Evolution Team

```json
{
  "name": "Standard Evolution",
  "require_vision_for_design": false,
  "members": [
    {
      "llm": "claude-3-opus",
      "role": "blue",
      "count": 3
    },
    {
      "llm": "gpt-4-turbo",
      "role": "red",
      "count": 2
    },
    {
      "llm": "claude-3-sonnet",
      "role": "judge",
      "count": 1
    }
  ]
}
```

#### Web Design Team (with vLLMs)

```json
{
  "name": "Web Design Team",
  "require_vision_for_design": true,
  "members": [
    {
      "llm": "gpt-4-vision",
      "role": "blue",
      "count": 2
    },
    {
      "llm": "claude-3-opus",
      "role": "blue",
      "count": 1
    },
    {
      "llm": "claude-3-sonnet",
      "role": "judge",
      "count": 1
    }
  ]
}
```

#### Code Review Team

```json
{
  "name": "Code Review",
  "require_diverse_providers": true,
  "members": [
    {
      "llm": "deepseek-coder",
      "role": "blue",
      "count": 2
    },
    {
      "llm": "gpt-4",
      "role": "red",
      "count": 2
    },
    {
      "llm": "claude-3-opus",
      "role": "judge",
      "count": 1
    }
  ]
}
```

---

## API Usage

### 1. Get Available LLMs

```bash
# All LLMs
GET /api/teams/llms/catalog

# Filtered
GET /api/teams/llms/catalog?provider=openai
GET /api/teams/llms/catalog?capability=vision
GET /api/teams/llms/catalog?vision_only=true
```

**Response**:
```json
{
  "llms": [...],
  "grouped": {
    "Vision/Multimodal (vLLM)": [
      {
        "provider": "openai",
        "model_id": "gpt-4-vision-preview",
        "name": "GPT-4 Vision",
        "is_vision": true,
        "capabilities": ["text", "vision"]
      }
    ],
    "Code Generation": [
      {
        "provider": "deepseek",
        "model_id": "deepseek-coder",
        "name": "DeepSeek Coder",
        "is_vision": false,
        "capabilities": ["text", "code"]
      }
    ]
  },
  "total": 15
}
```

### 2. Create Team

```bash
POST /api/teams/teams
{
  "name": "My Custom Team",
  "description": "Team with mixed providers",
  "voting_strategy": "consensus",
  "quorum_threshold": 0.7,
  "require_vision_for_design": true,
  "require_diverse_providers": true,
  "members": [
    {
      "member_id": "member_1",
      "llm": {
        "provider": "anthropic",
        "model_id": "claude-3-opus-20240229",
        "name": "Claude 3 Opus",
        "is_vision": true,
        "capabilities": ["text", "vision", "code"]
      },
      "role": "blue",
      "temperature": 0.7,
      "max_tokens": 4096,
      "credential_id": "bubblelab_123"
    }
  ]
}
```

### 3. Verify Credential

```bash
POST /api/teams/credentials/verify
{
  "provider": "openai",
  "api_key": "sk-...",
  "model_to_test": "gpt-3.5-turbo"
}
```

### 4. Quick Assignment

```bash
POST /api/teams/teams/assign
{
  "team_id": "team_123",
  "llm_provider": "anthropic",
  "llm_model_id": "claude-3-opus-20240229",
  "role": "blue",
  "temperature": 0.7
}
```

---

## Frontend Integration

### TypeScript Client

```typescript
import { teamAssignmentApi } from '@/services/teamAssignmentApi';

// Get LLM catalog
const catalog = await teamAssignmentApi.getLLMCatalog({
  vision_only: true  // Only vLLMs
});

// Create team
const team = await teamAssignmentApi.createTeam({
  name: 'My Team',
  members: [...],
  ...DEFAULT_TEAM_COMPOSITION
});

// Verify credential
const verification = await teamAssignmentApi.verifyCredential({
  provider: LLMProvider.OPENAI,
  api_key: 'sk-...',
  api_base: 'https://api.openai.com/v1'
});

if (verification.verified) {
  console.log('Credential works!');
}
```

### UI Components

#### LLM Selector (with vLLM grouping)

```typescript
// Separate sections in dropdown
const LLMSelector = () => {
  const [catalog, setCatalog] = useState<LLMSearchResponse | null>(null);

  useEffect(() => {
    teamAssignmentApi.getLLMCatalog().then(setCatalog);
  }, []);

  return (
    <Select>
      <SelectItem disabled>── Vision Models (vLLM) ──</SelectItem>
      {catalog?.vision_llms.map(llm => (
        <SelectItem key={llm.model_id} value={llm.model_id}>
          👁️ {llm.name}
        </SelectItem>
      ))}

      <SelectItem disabled>── Text Models ──</SelectItem>
      {catalog?.text_llms.map(llm => (
        <SelectItem key={llm.model_id} value={llm.model_id}>
          {llm.name}
        </SelectItem>
      ))}
    </Select>
  );
};
```

#### Team Builder

```typescript
const TeamBuilder = () => {
  const [members, setMembers] = useState<TeamMemberLLM[]>([]);

  const addMember = (llm: LLMModel, role: TeamRole) => {
    setMembers([...members, {
      member_id: `member_${Date.now()}`,
      llm,
      role,
      temperature: 0.7,
      max_tokens: 4096,
      total_requests: 0,
      successful_requests: 0,
    }]);
  };

  return (
    <div>
      <LLMSelector onSelect={(llm) => addMember(llm, TeamRole.BLUE_TEAM)} />
      <TeamMemberList members={members} />
      <CreateTeamButton members={members} />
    </div>
  );
};
```

---

## vLLM Handling

### Visual Separation

vLLMs are **clearly distinguished** in the UI:

```typescript
// In dropdowns and selectors
const vllmBadge = llm.is_vision ? (
  <Badge variant="vision">👁️ vLLM</Badge>
) : null;

// Or separate sections
{catalog.vision_llms.map(llm => (
  <VisionLLMCard llm={llm} />
))}
{catalog.text_llms.map(llm => (
  <TextLLMCard llm={llm} />
))}
```

### Automatic Routing

When a task requires vision capabilities:

```python
# Backend automatically selects vLLM
if task_requires_vision():
    vision_members = [m for m in team.members if m.llm.is_vision]
    if not vision_members:
        raise Error("Task requires vision but team has no vLLM")
    return vision_members[0]
```

### Recommended vLLMs for Design Tasks

| Model | Provider | Best For |
|-------|----------|----------|
| **GPT-4 Vision** | OpenAI | General UI design, mockups |
| **Claude 3 Sonnet** | Anthropic | Web design, screenshots |
| **Gemini Pro Vision** | Google | Document layouts, diagrams |
| **LlaVA-Next** | Custom (vLLM) | Open-source alternative |

---

## Examples

### Example 1: Create Design Team with vLLMs

```typescript
// 1. Get vLLMs
const { vision_llms } = await teamAssignmentApi.getVisionLLMs();

// 2. Pick GPT-4 Vision for blue team
const gpt4Vision = vision_llms.find(llm => llm.model_id === 'gpt-4-vision-preview');

// 3. Create team
const team = await teamAssignmentApi.createTeam({
  name: 'Web Design Team',
  description: 'Creates web designs with visual understanding',
  require_vision_for_design: true,
  members: [
    {
      member_id: 'blue_1',
      llm: gpt4Vision,
      role: TeamRole.BLUE_TEAM,
      temperature: 0.8, // Higher creativity
      max_tokens: 4096,
    },
    // Add more members...
  ],
  voting_strategy: 'consensus',
  quorum_threshold: 0.7,
});

console.log(`Team created: ${team.team_id}`);
```

### Example 2: Assign Different LLMs to Different Roles

```typescript
// Same LLM, different roles
const claude = await getLLM('claude-3-opus-20240229');

await teamAssignmentApi.addTeamMember(teamId, {
  llm: claude,
  role: TeamRole.BLUE_TEAM, // Generates
});

await teamAssignmentApi.addTeamMember(teamId, {
  llm: claude,
  role: TeamRole.RED_TEAM, // Attacks
});

await teamAssignmentApi.addTeamMember(teamId, {
  llm: claude,
  role: TeamRole.JUDGE, // Evaluates
});
```

### Example 3: Use Custom vLLM

```typescript
// 1. Add credential for local vLLM
const verification = await teamAssignmentApi.verifyCredential({
  provider: LLMProvider.OPENAI_LIKE,
  api_key: 'dummy-key', // vLLM might not need real key
  api_base: 'http://localhost:8000/v1', // vLLM server
  model_to_test: 'llava-next',
});

if (verification.verified) {
  // 2. Use in team
  const team = await teamAssignmentApi.createTeam({
    name: 'Local vLLM Team',
    members: [
    {
      member_id: 'local_1',
      llm: {
        provider: LLMProvider.OPENAI_LIKE,
        model_id: 'llava-next',
        name: 'LlaVA-Next (Local)',
        is_vision: true,
        capabilities: [LLMCapability.VISION, LLMCapability.TEXT],
        max_tokens: 4096,
        supports_streaming: true,
        supports_function_calling: false,
      },
      role: TeamRole.BLUE_TEAM,
      temperature: 0.7,
      max_tokens: 4096,
      credential_id: verification.credential_id,
    }
    ],
  });
}
```

### Example 4: Load Credentials from BubbleLab

```typescript
// List all credentials
const { credentials } = await teamAssignmentApi.listCredentials();

// Group by source
const bubblelabCreds = credentials.filter(
  c => c.source === CredentialSource.BUBBLELAB_CREDENTIALS
);
const configCreds = credentials.filter(
  c => c.source === CredentialSource.OPENEVOLVE_CONFIG
);

console.log(`${bubblelabCreds.length} from BubbleLab`);
console.log(`${configCreds.length} from config`);
```

---

## Best Practices

### 1. Always Verify Credentials

Before using in production, verify:

```typescript
const verification = await teamAssignmentApi.verifyCredential({
  provider: LLMProvider.OPENAI,
  api_key: userProvidedKey,
  model_to_test: 'gpt-3.5-turbo',
});

if (!verification.verified) {
  alert('Invalid credential');
  return;
}
```

### 2. Use vLLMs for Visual Tasks

For web design, UI generation, screenshot analysis:

```typescript
if (taskType === 'web-design' || taskType === 'ui-generation') {
  const visionTeam = await teamAssignmentApi.createTeam({
    name: 'Design Team',
    require_vision_for_design: true, // Enforce vLLM
    members: [...],
  });
}
```

### 3. Diverse Providers for Resilience

Avoid single provider dependency:

```typescript
const diverseTeam = {
  require_diverse_providers: true, // Enforce diversity
  members: [
    { llm: claude, provider: 'anthropic' },
    { llm: gpt4, provider: 'openai' },
    { llm: gemini, provider: 'google' },
  ],
};
```

### 4. Save Credentials in BubbleLab

For production use, save in BubbleLab credentials tab:

- ✅ Persistent storage
- ✅ Encrypted
- ✅ Team sharing
- ✅ Usage tracking

### 5. Use Team Templates

Quick start with predefined templates:

```typescript
const { templates } = await teamAssignmentApi.getTeamTemplates();
const webDesignTemplate = templates.find(t => t.id === 'web_design');

const team = await teamAssignmentApi.createTeamFromTemplate(
  'web_design',
  'My Web Design Team'
);
```

---

## Troubleshooting

### Issue: "No credential found"

**Solution**:
1. Check OpenEvolve `.env` file has API keys
2. Or add credentials in BubbleLab credentials tab
3. Verify credentials before using

### Issue: "Team requires at least one vLLM"

**Solution**:
```typescript
// Add a vLLM member
await teamAssignmentApi.addTeamMember(teamId, {
  llm: gpt4Vision, // or claude-3-opus
  role: TeamRole.BLUE_TEAM,
});
```

### Issue: "Verification failed"

**Solution**:
- Check API key is correct
- Check `api_base` URL (for custom providers)
- Check network/firewall settings

---

## API Endpoints Reference

### LLM Catalog
- `GET /api/teams/llms/catalog` - Get all LLMs
- `GET /api/teams/llms/catalog?vision_only=true` - Get only vLLMs
- `GET /api/teams/llms/providers` - Get supported providers

### Credentials
- `GET /api/teams/credentials` - List all credentials
- `POST /api/teams/credentials/verify` - Verify credential
- `POST /api/teams/credentials` - Add new credential

### Teams
- `POST /api/teams/teams` - Create team
- `GET /api/teams/teams` - List teams
- `GET /api/teams/teams/{id}` - Get team
- `PUT /api/teams/teams/{id}` - Update team
- `DELETE /api/teams/teams/{id}` - Delete team
- `POST /api/teams/teams/{id}/members` - Add member
- `DELETE /api/teams/teams/{id}/members/{member_id}` - Remove member
- `POST /api/teams/teams/assign` - Quick assign LLM to team

### Templates
- `GET /api/teams/teams/templates` - Get team templates
- `POST /api/teams/teams/templates/{id}/create` - Create from template

---

## Summary

This system provides:

✅ **Maximum Flexibility**: Any LLM to any team
✅ **Unified Credentials**: Single source of truth
✅ **vLLM Support**: Vision models clearly distinguished
✅ **Easy to Use**: Team templates and quick assignment
✅ **Production Ready**: Verification, error handling, logging

**Next**: Integrate with workflow engines (Evolution, Adversarial, Sovereign)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-27
**Status**: ✅ **IMPLEMENTED**
