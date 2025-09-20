# OpenEvolve Protocol Improver Frontend

A powerful Streamlit application for improving protocols and standard operating procedures (SOPs) through evolutionary refinement and adversarial testing using 34+ LLM providers.

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

1. **OpenRouter Setup**: Enter your OpenRouter API key to access 100+ models
2. **Select Teams**: 
   - **Red Team**: Models that identify flaws and vulnerabilities
   - **Blue Team**: Models that patch and fix identified issues
3. **Configure Testing**: Set confidence thresholds, iteration limits, and model parameters
4. **Start Testing**: Begin the adversarial improvement process

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
