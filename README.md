# OpenEvolve Protocol Improver

A powerful Streamlit UI Frontend for [OpenEvolve](https://github.com/codelion/openevolve) for improving protocols and standard operating procedures (SOPs), code and any other desired material through evolutionary refinement and adversarial testing using 34+ LLM providers.

## Features

### Dual Improvement Approaches
- **Evolution Mode**: Uses a single LLM provider to iteratively refine protocols through evolutionary improvement
- **Adversarial Testing**: Employs multiple LLM providers in a red team/blue team approach to identify vulnerabilities and generate hardened protocols

### Comprehensive LLM Support
- **34+ Providers**: OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Perplexity, Groq, Together AI, Fireworks, and many more
- **OpenRouter Integration**: Access to 100+ models through a single API for adversarial testing
- **Local Models**: Support for Ollama, LM-Studio, vLLM, and custom endpoints

### Advanced Features
- Real-time protocol evaluation and improvement
- Cost estimation and token tracking
- Thread-safe operations for concurrent model evaluation
- Comprehensive logging and result visualization
- Deterministic testing with seed support
- Per-model configuration for fine-tuned control

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd openevolve-protocol-improver
```

2. Install dependencies:
```bash
pip install streamlit requests streamlit-tags
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Evolution Mode

1. **Configure Provider**: Select from 34+ supported providers in the sidebar
2. **Set API Key**: Enter your API key or set the corresponding environment variable
3. **Paste Protocol**: Enter your draft protocol in the text area
4. **Configure Parameters**: Adjust temperature, max tokens, iterations, etc.
5. **Start Evolution**: Click "Start Evolution" to begin iterative improvement

### Adversarial Testing

The adversarial testing mode represents the core innovation of OpenEvolve, implementing a sophisticated multi-agent system that subjects protocols to rigorous security analysis and iterative hardening.

#### The Red Team/Blue Team Methodology

**Red Team (Critics)**
- Multiple diverse LLMs act as adversarial security auditors
- Each model independently analyzes the SOP for vulnerabilities, logical gaps, ambiguities, and potential abuse vectors
- Uses specialized prompting to encourage "uncompromising" analysis that identifies:
  - Edge cases and undefined responsibilities
  - Missing preconditions and unsafe defaults
  - Paths for malicious exploitation
  - Regulatory compliance gaps
  - Operational failure modes

**Blue Team (Defenders)**
- Different set of LLMs act as security engineers focused on remediation
- Receive the aggregated critique from red team models
- Generate comprehensive patches that address identified vulnerabilities
- Implement security principles like least privilege, fail-safe defaults, and defense in depth
- Add missing elements: preconditions, acceptance tests, rollback procedures, monitoring, and incident response

#### Advanced Consensus Mechanisms

**Intelligent Patch Selection**
- Evaluates multiple blue team solutions based on coverage metrics
- Scores patches by number of issues resolved vs. mitigated vs. left as residual risks
- Selects optimal solution using multi-criteria decision making
- Handles conflicting recommendations through automated arbitration

**Dynamic Risk Assessment**
- Aggregates findings across all red team models with severity weighting
- Tracks risk categories and calculates aggregate risk scores
- Monitors for issue clustering in specific protocol areas
- Provides quantitative risk metrics for decision making

#### Convergence and Quality Assurance

**Approval Voting System**
- All red team models vote on the final hardened protocol
- Uses configurable confidence thresholds (typically 85-95%)
- Prevents premature termination while avoiding infinite loops
- Tracks approval rates and average quality scores across iterations

**Stagnation Detection**
- Monitors protocol changes using content hashing
- Identifies when improvements have plateaued
- Suggests parameter adjustments when progress stalls
- Prevents wasted computation on converged solutions

### How Adversarial Testing Works

#### Phase 1: Red Team Analysis
Each red team model independently receives the current SOP and performs comprehensive security analysis:

```
Input: Current SOP + Red Team Critique Prompt
Output: Structured JSON with vulnerabilities classified by:
- Severity (low/medium/high/critical)
- Category (authentication, authorization, data handling, etc.)
- Exploitation paths and attack vectors
- Specific remediation recommendations
```

**Example Red Team Findings:**
- **Critical**: "Authentication bypass possible through parameter manipulation in step 3"
- **High**: "No rollback procedure defined for failed deployments" 
- **Medium**: "Ambiguous role definitions could lead to privilege escalation"
- **Low**: "Missing audit logging for configuration changes"

#### Phase 2: Blue Team Remediation
Blue team models receive both the original SOP and the aggregated red team critiques:

```
Input: Original SOP + All Red Team Critiques
Output: Complete rewritten SOP with:
- Point-by-point issue remediation
- Added security controls and guardrails
- Enhanced procedures and checkpoints
- Mitigation matrix tracking issue resolution
```

**Blue Team Improvements:**
- Explicit preconditions and validation steps
- Fail-safe defaults and error handling
- Comprehensive audit trails
- Role-based access controls
- Incident response procedures
- Rollback and recovery mechanisms

#### Phase 3: Consensus Building and Selection
The system evaluates multiple blue team patches using sophisticated scoring:

```python
# Scoring Algorithm (simplified)
coverage_score = (resolved_issues × 2) + (mitigated_issues × 1)
penalty = residual_risks × 2
final_score = coverage_score - penalty + quality_factors

# Selection considers:
- Issue resolution completeness
- Implementation feasibility  
- Operational impact
- Compliance alignment
```

#### Phase 4: Validation and Approval
The improved SOP undergoes final validation by the full red team:

```
Each red team model votes: APPROVED/REJECTED + 0-100 score
Confidence calculation: (approved_votes / total_votes) × 100%
Quality metric: average_score across all evaluations

Iteration continues until: confidence >= threshold (e.g., 90%)
```

#### Advanced Features and Controls

#### Advanced Team Rotation and Configuration

OpenEvolve supports sophisticated multi-LLM rotation strategies that maximize analytical diversity and prevent convergence bias:

**Rotation Strategies**

1. **Round Robin Rotation**
   ```python
   # Example: 5 red team models cycle through each iteration
   Iteration 1: [GPT-4, Claude-3, Gemini-Pro, Command-R, Llama-3]
   Iteration 2: [Claude-3, Gemini-Pro, Command-R, Llama-3, GPT-4]
   Iteration 3: [Gemini-Pro, Command-R, Llama-3, GPT-4, Claude-3]
   # Ensures each model gets equal participation
   ```

2. **Performance-Based Rotation**
   ```python
   # Models rotate based on vulnerability detection rates
   High performers: More frequent participation
   Lower performers: Reduced but maintained presence
   Adaptive weighting: Real-time adjustment based on contribution quality
   ```

3. **Random Sampling**
   ```python
   # Randomly select subset from larger pool each iteration
   Pool: 10 available red team models
   Active per iteration: 3-5 models (configurable)
   Prevents predictable patterns, maintains unpredictability
   ```

4. **Threshold-Triggered Rotation**
   ```python
   # Change team composition based on confidence metrics
   if confidence_plateau > 2_iterations:
       rotate_underperforming_models()
   if critical_issues_found:
       activate_specialist_security_models()
   ```

5. **Staged Rotation Patterns**
   ```python
   # Pre-defined rotation sequences
   Pattern A: [Security-focused → General → Compliance → Domain-expert]
   Pattern B: [Diverse-architectures → Similar-models → Hybrid-mix]
   Pattern C: [Conservative → Aggressive → Balanced → Validation]
   ```

**Multi-Tier Configuration System**

**Tier 1: Core Team Configuration**
```yaml
red_team:
  primary_models: [gpt-4, claude-3-opus, gemini-pro]
  rotation_strategy: "round_robin"
  rotation_interval: 2  # iterations
  min_active: 3
  max_active: 5

blue_team:
  primary_models: [command-r-plus, llama-3-70b, mixtral-8x7b]
  rotation_strategy: "performance_based"
  specialist_triggers: ["critical_security", "compliance_gap"]
```

**Tier 2: Advanced Rotation Logic**
```yaml
rotation_rules:
  stagnation_threshold: 3  # iterations without improvement
  confidence_plateau_trigger: 2  # iterations at same confidence
  underperformer_timeout: 5  # iterations before rotation
  specialist_activation:
    security_focused: [claude-3-opus, gpt-4-turbo]
    compliance_expert: [command-r-plus, gemini-pro]
    domain_specific: [custom_model_pool]
```

**Tier 3: Dynamic Adaptation**
```yaml
adaptive_features:
  performance_tracking: true
  real_time_adjustment: true
  cost_optimization: true
  quality_weighting: true
  
performance_metrics:
  vulnerability_detection_rate: weight=0.4
  solution_quality_score: weight=0.3
  cost_efficiency: weight=0.2
  response_reliability: weight=0.1
```

**Configuration Examples**

**Enterprise Security Audit Setup**
```python
config = {
    "red_team": {
        "models": ["gpt-4-turbo", "claude-3-opus", "gemini-pro", "command-r-plus", "llama-3-70b"],
        "strategy": "security_focused_rotation",
        "rotation_pattern": [
            # Round 1-2: Broad security analysis
            ["gpt-4-turbo", "claude-3-opus", "gemini-pro"],
            # Round 3-4: Compliance and regulatory focus  
            ["command-r-plus", "gemini-pro", "gpt-4-turbo"],
            # Round 5+: Specialized penetration testing mindset
            ["claude-3-opus", "llama-3-70b", "gpt-4-turbo"]
        ],
        "confidence_threshold_per_round": [70, 80, 90],
        "specialist_activation": {
            "critical_found": "activate_all_security_models",
            "compliance_gap": "add_regulatory_specialists"
        }
    }
}
```

**Cost-Optimized Configuration**
```python
config = {
    "budget_constraints": {
        "max_cost_per_iteration": 5.00,  # USD
        "total_budget_limit": 50.00,     # USD
        "cost_optimization": true
    },
    "rotation_strategy": {
        "primary": "cost_performance_ratio",
        "high_value_models": ["gpt-4-turbo", "claude-3-opus"],  # Use sparingly
        "workhouse_models": ["gpt-3.5-turbo", "llama-3-8b"],   # Use frequently
        "rotation_logic": "cost_weighted_round_robin"
    }
}
```

**Research and Development Setup**
```python
config = {
    "experimental_features": {
        "model_ensemble_voting": true,
        "cross_architecture_validation": true,
        "novel_model_integration": true
    },
    "rotation_strategies": [
        "random_forest_sampling",      # ML-inspired selection
        "genetic_algorithm_evolution", # Evolutionary model selection
        "reinforcement_learning_adaptation", # Self-improving team composition
        "diversity_maximization"       # Maximize analytical perspectives
    ]
}
```

**Real-Time Adaptation Features**

**Performance Monitoring Dashboard**
- Live tracking of model contribution quality
- Real-time cost vs. value analysis  
- Vulnerability detection effectiveness metrics
- Solution implementation success rates

**Intelligent Rotation Triggers**
- Automatic underperformer replacement
- Specialist model activation for complex issues
- Budget-aware model substitution
- Quality-driven team composition adjustments

**Predictive Team Optimization**
- ML-based prediction of optimal team compositions
- Historical performance analysis for model selection
- Workload-appropriate model matching
- Domain-specific expertise routing

This multi-layered configuration system ensures that teams can be precisely tuned for specific use cases while maintaining the flexibility to adapt in real-time based on performance metrics and emerging requirements.

**Enterprise Integration**
- Export results in multiple formats (JSON, PDF, Word)
- Integration APIs for CI/CD pipelines
- Compliance reporting and audit trails
- Version control integration for protocol evolution tracking

#### Output and Analysis

**Comprehensive Results Package**
- Final hardened protocol with full traceability
- Iteration-by-iteration improvement history
- Risk assessment matrices and vulnerability classifications
- Model-specific contributions and voting patterns
- Cost breakdown and token utilization metrics

**Quality Metrics**
- Approval rate progression across iterations
- Issue resolution rates by category and severity
- Model agreement/disagreement analysis
- Cost-effectiveness ratios

## Supported Providers

### Major Cloud Providers
- OpenAI (GPT-4, GPT-3.5, etc.)
- Azure OpenAI
- Anthropic (Claude models)
- Google Gemini
- AWS Bedrock (Claude, Titan, Cohere, Jurassic, Llama)
- Vertex AI

### Specialized Providers
- Mistral AI
- Cohere
- Perplexity
- Groq
- Together AI
- Fireworks
- Replicate
- Anyscale
- DeepInfra
- OctoAI

### Chinese Providers
- Moonshot
- Baichuan
- Zhipu AI
- MiniMax
- Yi (LingyiWanwu)
- DeepSeek

### Local/Self-Hosted
- Ollama
- LM-Studio
- vLLM
- SageMaker
- Databricks

### Aggregators
- OpenRouter
- Custom endpoints

## Environment Variables

Set these environment variables for automatic API key detection:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"
export COHERE_API_KEY="your-key"
export PERPLEXITY_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
# ... and many more
```

## Configuration

### Evolution Parameters
- **Max Iterations**: Maximum number of improvement cycles
- **Temperature**: Controls randomness in model responses
- **Top-p**: Nucleus sampling parameter
- **Frequency/Presence Penalty**: Controls repetition
- **Seed**: For deterministic results
- **System Prompt**: Custom instructions for the improvement process

### Adversarial Testing Parameters
- **Confidence Threshold**: Percentage of red team approval needed to stop
- **Min/Max Iterations**: Bounds for the testing process
- **Max Tokens**: Token limit per model response
- **Max Workers**: Number of parallel model requests
- **Force JSON**: Enforce structured JSON responses
- **Per-Model Config**: Individual temperature, top-p, etc. for each model

## How It Works

### Evolution Mode
1. Takes your draft protocol as input
2. Iteratively refines it using the selected LLM
3. Applies your custom system prompt for specific improvement goals
4. Tracks progress and provides real-time updates
5. Outputs the improved protocol

### Adversarial Testing
1. **Red Team Phase**: Multiple models analyze the SOP for vulnerabilities
2. **Blue Team Phase**: Different models create patches to fix identified issues
3. **Consensus Building**: Selects the best patch based on coverage and quality
4. **Approval Check**: Red team models vote on the improved SOP
5. **Iteration**: Process repeats until confidence threshold is met

## Cost Estimation

The application provides real-time cost estimates based on:
- Token usage (prompt + completion)
- Provider-specific pricing
- Model-specific rates
- Total spend across all models and iterations

## Thread Safety

- All operations are thread-safe with proper locking
- Multiple models can be queried concurrently
- Real-time UI updates without blocking
- Graceful error handling and recovery

## Limitations

- Some providers have non-OpenAI compatible APIs (noted in Evolution mode)
- Rate limits vary by provider
- Context window limits may affect large protocols
- Network connectivity required for cloud providers

## Error Handling

- Exponential backoff for transient errors
- Automatic retry logic with jitter
- Graceful degradation for failed models
- Comprehensive error logging
- User-friendly error messages

## Security Considerations

- API keys are handled securely
- No logging of sensitive information
- Optional deterministic seeds for reproducibility
- Configurable rate limiting and timeouts


- Built with Streamlit for the web interface
- Uses the requests library for HTTP communication
- Supports 34+ LLM providers for maximum flexibility
- Inspired by evolutionary algorithms and adversarial testing methodologies
