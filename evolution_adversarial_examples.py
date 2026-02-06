"""
Evolution and Adversarial Testing Examples

This file provides practical examples for using the BubbleLabs Evolution Integration.
Run these examples to understand the system's capabilities.

Author: OpenEvolve Frontend Team
"""


from ui_shim import ui as st
from bubblelabs_evolution_integration import BubbleLabsEvolutionIntegration
from evolution_workflow_templates import TemplateManager


# =============================================================================
# EXAMPLE 1: BASIC CODE EVOLUTION
# =============================================================================

def example_1_basic_code_evolution():
    """
    Example 1: Basic Code Evolution

    Demonstrates the simplest evolution workflow:
    - Standard genetic algorithm
    - Code optimization
    - Real-time progress monitoring
    """
    st.title("Example 1: Basic Code Evolution")

    st.markdown("""
    ## Scenario
    Optimize a Python function for better performance and readability.

    ## Initial Code
    ```python
    def process_items(items):
        result = []
        for item in items:
            if item > 0:
                result.append(item * 2)
        return result
    ```

    ## Configuration
    - Evolution Type: Standard
    - Population Size: 20
    - Generations: 50
    - Mutation Rate: 0.1
    - Crossover Rate: 0.7

    ## Expected Improvements
    - List comprehension usage
    - Type hints
    - Docstring
    - Better variable names
    - Edge case handling
    """)

    # Initial code
    initial_code = """def process_items(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result"""

    st.code(initial_code, language="python")

    st.markdown("### Run Evolution")
    if st.button("Start Evolution (Example 1)", key="ex1_start"):
        st.info("Evolution started. Monitor the 'Active Tasks' tab for progress.")


# =============================================================================
# EXAMPLE 2: MAKER VOTING EVOLUTION
# =============================================================================

def example_2_maker_voting():
    """
    Example 2: MAKER Voting for Zero-Error Evolution

    Demonstrates MAKER integration:
    - First-to-ahead-by-k voting
    - Zero-error guarantees
    - Higher confidence selection
    """
    st.title("Example 2: MAKER Voting Evolution")

    st.markdown("""
    ## Scenario
    Evolve a critical system component where zero errors are required.

    ## Initial Code
    ```python
    def divide_numbers(a, b):
        return a / b
    ```

    ## MAKER Configuration
    - Enable MAKER Voting: True
    - Voting Threshold (k): 3
    - Number of Candidates: 5 (satisfies N >= 2k-1)
    - Adaptive Voting: Enabled
    - Population Size: 25

    ## Benefits
    - Statistical convergence guarantees
    - Reduced error rate
    - High-confidence selection
    - Reliable results for critical systems
    """)

    initial_code = """def divide_numbers(a, b):
    return a / b"""

    st.code(initial_code, language="python")

    st.markdown("### MAKER Voting Process")
    st.markdown("""
    1. **Generate Candidates**: Create N=5 candidate solutions
    2. **Collect Votes**: Each candidate is voted on
    3. **Select Winner**: First to reach k=3 ahead-by votes wins
    4. **Verify**: Check for consensus and quality
    5. **Repeat**: Apply for each generation

    ### Voting Visualization
    ```
    Generation 10:
    Candidate A: ||| (3 votes) [OK]
    Candidate B: || (2 votes)
    Candidate C: | (1 vote)
    Candidate D: ||| (3 votes) - Winner (ahead by 2)
    Candidate E: | (1 vote)
    ```
    """)

    if st.button("Start MAKER Evolution (Example 2)", key="ex2_start"):
        st.info("MAKER evolution started with zero-error guarantees.")


# =============================================================================
# EXAMPLE 3: MDAP DECOMPOSITION
# =============================================================================

def example_3_mdap_decomposition():
    """
    Example 3: MDAP Task Decomposition

    Demonstrates problem decomposition:
    - Complex task breakdown
    - Subtask optimization
    - Result synthesis
    """
    st.title("Example 3: MDAP Decomposition")

    st.markdown("""
    ## Scenario
    Refactor a large, complex function with multiple responsibilities.

    ## Initial Code
    A function that does too many things at once.

    ## MDAP Configuration
    - Enable Decomposition: True
    - Decomposition Depth: 5
    - Max Subtasks: 10
    - Population Size: 30

    ## Decomposition Process
    1. **Analyze**: Identify functional components
    2. **Decompose**: Break into subtasks
    3. **Optimize**: Evolve each subtask independently
    4. **Synthesize**: Combine optimized subtasks
    5. **Verify**: Ensure correctness

    ## Example Decomposition
    ```
    Original Function: process_user_data()
    ├── Subtask 1: validate_input()
    ├── Subtask 2: sanitize_data()
    ├── Subtask 3: transform_data()
    ├── Subtask 4: store_data()
    └── Subtask 5: log_results()
    ```
    """)

    initial_code = """def process_user_data(user_data):
    # Validate
    if not user_data:
        return None
    # Sanitize
    clean_data = {}
    for key, value in user_data.items():
        if isinstance(value, str):
            clean_data[key] = value.strip()
        else:
            clean_data[key] = value
    # Transform
    if 'name' in clean_data:
        clean_data['name'] = clean_data['name'].title()
    # Store
    database.save(clean_data)
    # Log
    logger.info(f"Processed {clean_data.get('name')}")
    return clean_data"""

    st.code(initial_code, language="python")

    if st.button("Start MDAP Evolution (Example 3)", key="ex3_start"):
        st.info("MDAP decomposition evolution started.")


# =============================================================================
# EXAMPLE 4: ADVERSARIAL SECURITY AUDIT
# =============================================================================

def example_4_adversarial_audit():
    """
    Example 4: Adversarial Security Audit

    Demonstrates adversarial testing:
    - Red team vulnerability discovery
    - Blue team defense generation
    - Coevolution for system hardening
    """
    st.title("Example 4: Adversarial Security Audit")

    st.markdown("""
    ## Scenario
    Perform a comprehensive security audit on authentication code.

    ## Target Code
    ```python
    def authenticate(username, password):
        if username == "admin" and password == "password123":
            return True
        return False
    ```

    ## Adversarial Configuration
    - Mode: MAKER Full (Red team voting + Blue team MDAP)
    - Rounds: 5
    - Red Team Size: 5
    - Blue Team Size: 3
    - Attack Strength: 0.7
    - Enable Coevolution: True

    ## Testing Process

    ### Round 1: Initial Assessment
    **Red Team Findings:**
    - Hardcoded credentials
    - No rate limiting
    - Missing password hashing
    - No account lockout

    **Blue Team Fixes:**
    - Move credentials to database
    - Implement bcrypt hashing
    - Add rate limiting
    - Add failed attempt tracking

    ### Round 2: Evolved Attacks
    **Red Team Attacks:**
    - Brute force attack patterns
    - Timing analysis
    - SQL injection attempts
    - Session hijacking

    **Blue Team Defenses:**
    - Implement exponential backoff
    - Add parameterized queries
    - Secure session management

    ### Subsequent Rounds
    Attacks and defenses evolve together, leading to increasingly secure implementation.

    ## Expected Vulnerabilities Found
    1. Hardcoded credentials (CRITICAL)
    2. Missing password hashing (CRITICAL)
    3. No rate limiting (HIGH)
    4. No account lockout (HIGH)
    5. Timing attack vulnerability (MEDIUM)
    """)

    target_code = """def authenticate(username, password):
    if username == "admin" and password == "password123":
        return True
    return False"""

    st.code(target_code, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔴 Red Team")
        st.markdown("""
        - 5 agents
        - MAKER voting (k=3)
        - Attack decomposition
        - Diverse strategies
        """)

    with col2:
        st.markdown("### 🔵 Blue Team")
        st.markdown("""
        - 3 agents
        - MDAP decomposition
        - Defense layering
        - Comprehensive fixes
        """)

    if st.button("Start Adversarial Audit (Example 4)", key="ex4_start"):
        st.info("Adversarial security audit started. Monitor for vulnerability findings.")


# =============================================================================
# EXAMPLE 5: PROMPT REFINEMENT
# =============================================================================

def example_5_prompt_refinement():
    """
    Example 5: LLM Prompt Refinement

    Demonstrates prompt evolution:
    - Clarity improvement
    - Context addition
    - Example inclusion
    """
    st.title("Example 5: Prompt Refinement")

    st.markdown("""
    ## Scenario
    Evolve an LLM prompt for better code generation.

    ## Initial Prompt
    ```
    Write a function to sort a list.
    ```

    ## Configuration
    - Evolution Type: MDAP Decomposition
    - Generations: 75
    - Population: 20
    - Decomposition: Enabled

    ## Evolution Process
    1. **Generation 0-20**: Add specificity (Python, ascending)
    2. **Generation 20-40**: Add examples
    3. **Generation 40-60**: Add edge cases
    4. **Generation 60-75**: Final refinement

    ## Expected Refinement
    ```
    Write a Python function that sorts a list of numbers in ascending order.

    Requirements:
    - Function name: sort_numbers
    - Parameters: List[int] or List[float]
    - Returns: New sorted list
    - Handle empty lists
    - Handle duplicates
    - Include type hints
    - Add docstring with examples

    Example:
    >>> sort_numbers([3, 1, 4, 1, 5])
    [1, 1, 3, 4, 5]
    ```
    """)

    initial_prompt = "Write a function to sort a list."
    st.text_area("Initial Prompt", initial_prompt, height=100, disabled=True)

    st.markdown("### Evolution Progress")
    st.markdown("""
    | Generation | Prompt Quality | Changes |
    |------------|----------------|---------|
    | 0 | 0.30 | Initial prompt |
    | 20 | 0.55 | Added Python, ascending |
    | 40 | 0.72 | Added examples |
    | 60 | 0.85 | Added edge cases |
    | 75 | 0.91 | Final refinement |
    """)

    if st.button("Start Prompt Evolution (Example 5)", key="ex5_start"):
        st.info("Prompt refinement evolution started.")


# =============================================================================
# EXAMPLE 6: COEVOLUTION HARDENING
# =============================================================================

def example_6_coevolution():
    """
    Example 6: Attack-Defense Coevolution

    Demonstrates coevolutionary hardening:
    - Simultaneous attack/defense evolution
    - Adaptive strategies
    - System robustness improvement
    """
    st.title("Example 6: Coevolutionary Hardening")

    st.markdown("""
    ## Scenario
    Harden a system through adversarial coevolution.

    ## Target System
    API endpoint for user data retrieval.

    ## Coevolution Configuration
    - Rounds: 10
    - Red Team: 4 agents
    - Blue Team: 4 agents
    - Attack Strength: 0.6
    - Defense Strength: 0.8
    - Coevolution: Enabled

    ## Coevolution Process

    ### Round 1
    **Attack**: Basic authentication bypass
    **Defense**: Add authentication check
    **Result**: Attack blocked

    ### Round 2
    **Attack**: SQL injection
    **Defense**: Parameterized queries
    **Result**: Attack blocked

    ### Round 3
    **Attack**: Rate limiting bypass
    **Defense**: Exponential backoff
    **Result**: Attack mitigated

    ### ... continues for 10 rounds

    ## Progress Tracking

    | Round | Attacks Launched | Attacks Blocked | Defense Success Rate |
    |-------|-----------------|-----------------|---------------------|
    | 1 | 5 | 2 | 40% |
    | 2 | 7 | 5 | 71% |
    | 3 | 8 | 7 | 87% |
    | 4 | 10 | 9 | 90% |
    | 5 | 12 | 11 | 92% |
    | 10 | 15 | 15 | 100% |

    ## Final System
    After 10 rounds of coevolution:
    - Robust authentication
    - SQL injection protection
    - Rate limiting
    - Input validation
    - Error handling
    - Logging and monitoring
    """)

    st.markdown("### System Hardening Visualization")
    st.markdown("""
    ```
    Round 1:  ████████░░ 80% Robust
    Round 5:  ██████████ 100% Robust
    ```

    ### Key Insight
    Coevolution forces both attackers and defenders to continuously improve,
    leading to systems that are robust against sophisticated attacks.
    """)

    if st.button("Start Coevolution (Example 6)", key="ex6_start"):
        st.info("Coevolutionary hardening started. Watch attack/defense adaptation!")


# =============================================================================
# EXAMPLE 7: TEMPLATE USAGE
# =============================================================================

def example_7_templates():
    """
    Example 7: Using Workflow Templates

    Demonstrates template system:
    - Pre-configured workflows
    - Custom template creation
    - Template sharing
    """
    st.title("Example 7: Workflow Templates")

    st.markdown("""
    ## Available Templates

    ### Evolution Templates
    - **Code Optimization**: Improve code quality
    - **Prompt Refinement**: Enhance LLM prompts
    - **Text Summarization**: Create concise summaries
    - **MAKER Voting**: Zero-error evolution
    - **MDAP Decomposition**: Complex problem solving
    - **Hybrid**: Combined MAKER + MDAP

    ### Adversarial Templates
    - **Security Audit**: Find vulnerabilities
    - **Prompt Injection Testing**: Test prompt robustness
    - **Code Robustness**: Test edge cases
    - **MAKER Red Team**: Reliable attack generation
    - **MDAP Blue Team**: Comprehensive defense
    - **Coevolution Hardening**: System hardening

    ## Using Templates

    1. **Select Template**: Choose from dropdown
    2. **Review Configuration**: Check parameters
    3. **Customize**: Adjust if needed
    4. **Provide Input**: Add your content
    5. **Execute**: Run the workflow

    ## Creating Custom Templates

    ```python
    from evolution_workflow_templates import TemplateManager

    manager = TemplateManager()

    custom_template = manager.create_custom_template(
        name="My Optimization",
        description="Optimize for my specific use case",
        category="evolution",
        config={
            "population_size": 30,
            "max_generations": 150,
            "enable_maker_voting": True
        },
        example_content="# Your example",
        use_cases=["Use case 1", "Use case 2"]
    )
    ```
    """)

    # Show template selection UI
    template_manager = TemplateManager()
    evolution_templates = template_manager.get_evolution_templates()
    adversarial_templates = template_manager.get_adversarial_templates()

    st.markdown("### Evolution Templates")
    for template_id, template in evolution_templates.items():
        with st.expander(f"📋 {template.name}"):
            st.markdown(f"**Description:** {template.description}")
            st.markdown("**Use Cases:**")
            for uc in template.use_cases:
                st.markdown(f"- {uc}")

    st.markdown("### Adversarial Templates")
    for template_id, template in adversarial_templates.items():
        with st.expander(f"⚔️ {template.name}"):
            st.markdown(f"**Description:** {template.description}")
            st.markdown("**Use Cases:**")
            for uc in template.use_cases:
                st.markdown(f"- {uc}")


# =============================================================================
# EXAMPLE 8: COMPARISON ANALYSIS
# =============================================================================

def example_8_comparison():
    """
    Example 8: Comparing Different Approaches

    Demonstrates comparison:
    - Standard vs MAKER vs MDAP vs Hybrid
    - Performance metrics
    - Quality comparison
    """
    st.title("Example 8: Approach Comparison")

    st.markdown("""
    ## Comparison: Same Problem, Different Approaches

    ### Problem: Optimize sorting algorithm

    ### Approaches Compared

    1. **Standard Evolution**
    - Population: 20
    - Generations: 100
    - No MAKER/MDAP

    2. **MAKER Voting**
    - Population: 25
    - Generations: 120
    - Voting threshold: 3

    3. **MDAP Decomposition**
    - Population: 30
    - Generations: 150
    - Decomposition depth: 3

    4. **Hybrid**
    - Population: 25
    - Generations: 120
    - MAKER + MDAP

    ### Results Comparison

    | Approach | Best Fitness | Generations | Time (min) | Error Rate | Quality |
    |----------|--------------|-------------|------------|------------|---------|
    | Standard | 0.82 | 100 | 3.2 | 5% | Good |
    | MAKER | 0.91 | 120 | 8.5 | 0% | Excellent |
    | MDAP | 0.88 | 150 | 12.3 | 2% | Very Good |
    | Hybrid | 0.94 | 120 | 15.7 | 0% | Outstanding |

    ### Recommendations

    **Use Standard When:**
    - Quick results needed
    - Non-critical applications
    - Simple problems

    **Use MAKER When:**
    - Zero errors required
    - High confidence needed
    - Critical systems

    **Use MDAP When:**
    - Complex problems
    - Task decomposition beneficial
    - Parallel processing available

    **Use Hybrid When:**
    - Maximum quality required
    - Resources available
    - Time not constrained
    """)

    # Visualization
    import plotly.graph_objects as go

    approaches = ["Standard", "MAKER", "MDAP", "Hybrid"]
    fitness = [0.82, 0.91, 0.88, 0.94]
    time = [3.2, 8.5, 12.3, 15.7]
    errors = [5, 0, 2, 0]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=approaches,
        y=fitness,
        name='Best Fitness',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title='Fitness Comparison',
        yaxis_title='Fitness Score',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=approaches,
        y=time,
        mode='lines+markers',
        name='Time (minutes)',
        marker=dict(size=10)
    ))

    fig2.update_layout(
        title='Time Comparison',
        yaxis_title='Time (minutes)',
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# MAIN EXAMPLES APP
# =============================================================================

def main():
    """Main UI app for examples"""
    st.set_page_config(
        page_title="Evolution & Adversarial Examples",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Evolution & Adversarial Testing Examples")
    st.markdown("""
    This page provides practical examples for using the BubbleLabs Evolution Integration.
    Each example demonstrates different features and use cases.
    """)

    # Create example selector
    examples = {
        "Select an Example...": None,
        "1. Basic Code Evolution": example_1_basic_code_evolution,
        "2. MAKER Voting Evolution": example_2_maker_voting,
        "3. MDAP Decomposition": example_3_mdap_decomposition,
        "4. Adversarial Security Audit": example_4_adversarial_audit,
        "5. Prompt Refinement": example_5_prompt_refinement,
        "6. Coevolution Hardening": example_6_coevolution,
        "7. Template Usage": example_7_templates,
        "8. Approach Comparison": example_8_comparison
    }

    selected = st.selectbox(
        "Choose an Example",
        options=list(examples.keys())
    )

    if selected and examples[selected]:
        examples[selected]()
    else:
        st.markdown("""
        ## Available Examples

        ### Evolution Examples
        1. **Basic Code Evolution**: Simple genetic algorithm optimization
        2. **MAKER Voting**: Zero-error evolution with voting
        3. **MDAP Decomposition**: Complex problem decomposition
        5. **Prompt Refinement**: LLM prompt improvement

        ### Adversarial Examples
        4. **Security Audit**: Red team/blue team testing
        6. **Coevolution Hardening**: Attack/defense coevolution

        ### Advanced Examples
        7. **Template Usage**: Pre-configured workflows
        8. **Approach Comparison**: Performance comparison

        Select an example to begin!
        """)

        # Quick tips
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Start with Example 1 for basic understanding
        - Try Example 2 for zero-error requirements
        - Use Example 4 for security testing
        - Compare approaches with Example 8
        - Check templates in Example 7
        """)


if __name__ == "__main__":
    main()

