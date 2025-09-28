# OpenEvolve Frontend

The OpenEvolve Frontend is a comprehensive, general-purpose content improvement tool that provides a user-friendly graphical interface for the OpenEvolve backend. It supports various content types including protocols, code, documentation, and any other text-based content.

## 🌟 Key Features

### 🧬 AI-Powered Content Evolution
- **Iterative Improvement**: Uses evolutionary algorithms to continuously refine content
- **Multi-LLM Ensemble**: Leverages multiple AI models for diverse perspectives
- **Smart Evaluation**: Automatically assesses content quality and identifies improvements

### ⚔️ Adversarial Testing
- **Red Team/Blue Team**: Simulates attackers and defenders to harden content
- **Multi-Model Consensus**: Uses diverse AI models for robust testing
- **Confidence Tracking**: Monitors improvement through statistical confidence metrics

### 🌐 Universal Compatibility
- **Any Content Type**: Works with code, protocols, documentation, SOPs, policies, and more
- **Multi-LLM Support**: Compatible with OpenAI, Anthropic, Google, Mistral, and any OpenAI-compatible API
- **Flexible Integration**: Connects to GitHub, GitLab, Jira, Slack, and custom webhooks

### 👥 Collaboration & Sharing
- **Real-time Editing**: Multiple users can collaborate simultaneously
- **Version Control**: Complete history with branching and tagging
- **Commenting System**: Threaded discussions with mentions and notifications
- **Project Sharing**: Secure sharing with password protection

### 📊 Advanced Analytics
- **Performance Tracking**: Real-time metrics and progress visualization
- **Model Comparison**: Performance analysis across different AI models
- **Issue Classification**: Categorization by severity, type, and impact
- **Compliance Reporting**: Automatic compliance checking and reporting

### 🛠️ Developer Experience
- **Template Marketplace**: Extensive collection of pre-built templates
- **Export Options**: PDF, DOCX, HTML, JSON, LaTeX, and more
- **Custom Themes**: Light/dark mode with customizable appearance
- **Keyboard Shortcuts**: Productivity-enhancing hotkeys

## 🏗️ Architecture

### Dual-Mode Operation
1. **General Content Mode**: For protocols, documentation, and text content (primary focus)
2. **Code Evolution Mode**: Integration with OpenEvolve backend for specialized code improvements

### Component Structure
- **Frontend**: Streamlit-based UI with modular components
- **Backend Integration**: Direct API calls and OpenEvolve library integration
- **Storage**: In-memory session state with export capabilities
- **Analytics**: Real-time metrics and reporting engine

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/openevolve/openevolve.git
cd openevolve/frontend

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Streamlit application
streamlit run main.py
```

### First-Time Setup
1. Configure your LLM provider API key in the sidebar
2. Select an appropriate model for your use case
3. Load a template or enter your own content
4. Choose between Evolution or Adversarial Testing modes
5. Start improving your content!

## 📖 Usage Guide

### Content Evolution Mode
1. **Configure Provider**: Set up your LLM provider and API key
2. **Enter Content**: Paste your content or load a template
3. **Adjust Parameters**: Fine-tune evolution settings
4. **Start Process**: Click "Start Evolution" to begin
5. **Monitor Progress**: Watch real-time improvements
6. **Export Results**: Save your improved content

### Adversarial Testing Mode
1. **Select Models**: Choose red team (critics) and blue team (fixers)
2. **Configure Testing**: Set iterations, confidence thresholds, and parameters
3. **Run Testing**: Start adversarial testing process
4. **Review Results**: Analyze findings and improvements
5. **Generate Reports**: Export detailed test reports

### Collaboration Features
1. **Invite Teammates**: Share project links with collaborators
2. **Comment on Content**: Add notes and feedback
3. **Track Changes**: Monitor version history
4. **Resolve Issues**: Mark comments as resolved

## 🔧 Integration with OpenEvolve Backend

The frontend seamlessly integrates with the OpenEvolve backend for specialized code evolution:
- **Automatic Content Type Detection**: Routes code content to backend, general content to API
- **Enhanced Code Analysis**: Deep static analysis and optimization
- **Language-Specific Processing**: Specialized handling for Python, JavaScript, Java, and more
- **Performance Benchmarking**: Execution-based evaluation metrics

## 📈 Advanced Features

### AI Insights Dashboard
- **Quality Scoring**: Automated content quality assessment
- **Readability Analysis**: Sentence structure and complexity metrics
- **Structure Evaluation**: Organization and formatting assessment
- **Compliance Checking**: Regulatory and policy compliance analysis

### Machine Learning Integration
- **Protocol Suggestions**: AI-powered improvement recommendations
- **Content Classification**: Automatic categorization and tagging
- **Pattern Recognition**: Identification of common issues and best practices
- **Predictive Analytics**: Forecasting improvement potential

### Customization Options
- **Theme Settings**: Light/dark mode with custom colors
- **Layout Preferences**: Adjustable panels and views
- **Keyboard Shortcuts**: Customizable hotkeys
- **Export Templates**: Personalized export formats

## 🤝 Contributing

We welcome contributions from the community! Please see our contributing guidelines for details.

## 📄 License

Apache 2.0

---

*OpenEvolve: Making content smarter, safer, and stronger through AI-powered evolution.*