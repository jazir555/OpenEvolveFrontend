# ------------------------------------------------------------------
# 7. Main layout with tabs
# ------------------------------------------------------------------

st.markdown('<h2 style="text-align: center; color: #4a6fa5;">🧬 OpenEvolve Content Improver</h2>',
            unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem;">AI-Powered Content Hardening with Multi-LLM Consensus</p>',
    unsafe_allow_html=True)
st.markdown("---")

# Project information with enhanced UI
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("## 🔴🔵 Adversarial Testing & Evolution-based Content Improvement")
with col2:
    st.markdown(
        '<span class="team-badge-lg red-team">Red Team</span><span class="team-badge-lg blue-team">Blue Team</span>',
        unsafe_allow_html=True)
with col3:
    # Add a quick action button with enhanced styling
    if st.button("📋 Quick Guide", key="quick_guide_btn", use_container_width=True):
        st.session_state.show_quick_guide = not st.session_state.get("show_quick_guide", False)

# Show quick guide if requested with enhanced UI
if st.session_state.get("show_quick_guide", False):
    with st.expander("📘 Quick Guide", expanded=True):
        st.markdown("""
        ### 🚀 Getting Started

        1. **Choose Your Approach**:
           - **Evolution Tab**: Iteratively improve any content using one AI model
           - **Adversarial Testing Tab**: Harden content using multiple AI models in red team/blue team approach

        2. **Configure Your Models**:
           - Select a provider and model in the sidebar (Evolution tab)
           - Enter your OpenRouter API key for Adversarial Testing
           - Choose models for red team (critics) and blue team (fixers)

        3. **Input Your Content**:
           - Paste your existing content or load a template
           - Add compliance requirements if needed

        4. **Run the Process**:
           - Adjust parameters as needed
           - Click "Start" and monitor progress
           - Review results and save improved versions

        5. **Collaborate & Share**:
           - Add collaborators to your project
           - Save versions and track changes
           - Export results in multiple formats
        """)
        if st.button("Close Guide"):
            st.session_state.show_quick_guide = False
            st.rerun()

# Project info in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📁 Project Information")
    st.text_input("Project Name", key="project_name")
    st.text_area("Project Description", key="project_description", height=100)

    # Tags
    if HAS_STREAMLIT_TAGS:
        st.multiselect("Tags", st.session_state.tags, key="tags")

    # Collaborators
    st.multiselect("Collaborators", st.session_state.collaborators, key="collaborators")

    # Version control
    st.markdown("---")
    st.subheader("🔄 Version Control")
    if st.button("💾 Save Current Version"):
        if st.session_state.protocol_text.strip():
            version_name = st.text_input("Version Name", f"Version {len(st.session_state.protocol_versions) + 1}")
            comment = st.text_area("Comment", height=50)
            if st.button("✅ Confirm Save", key="confirm_save_version"):
                version_id = create_new_version(st.session_state.protocol_text, version_name, comment)
                st.success(f"✅ Version saved! ID: {version_id[:8]}")
                st.rerun()

    # Show version history with enhanced UI
    versions = get_version_history()
    if versions:
        st.write("### 📚 Version History")
        # Add version comparison feature
        version_options = [f"{v['name']} ({v['timestamp'][:10]})" for v in reversed(versions[-10:])]
        selected_versions = st.multiselect("Select versions to compare", version_options, key="version_comparison")

        # Show timeline view toggle
        show_timeline = st.toggle("Show Timeline View", key="show_timeline")

        if show_timeline:
            # Show visual timeline
            st.markdown(render_version_timeline(), unsafe_allow_html=True)
        else:
            # Show traditional list view with enhanced features
            for version in reversed(versions[-5:]):  # Show last 5 versions
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    if st.button(f"{version['name']}", key=f"load_version_{version['id']}", help=f"Load this version"):
                        load_version(version['id'])
                        st.success(f"✅ Loaded version: {version['name']}")
                        st.rerun()
                with col2:
                    st.caption(f"{version['timestamp'][:10]}")
                with col3:
                    st.caption(f"v{version['id'][:8]}")
                with col4:
                    # Add branch button for non-current versions
                    if version['id'] != st.session_state.get('current_version_id', ''):
                        if st.button("🌿", key=f"branch_icon_{version['id']}", help="Create a branch from this version"):
                            branch_name = st.text_input("Branch Name", f"Branch of {version['name']}",
                                                        key=f"branch_name_{version['id']}")
                            if st.button("✅ Create Branch", key=f"create_branch_{version['id']}"):
                                new_branch_id = branch_version(version['id'], branch_name)
                                if new_branch_id:
                                    st.success(f"✅ Branched to: {branch_name}")
                                    st.rerun()

    # Add collaborative workspace features
    st.markdown("---")
    st.subheader("👥 Collaborative Workspace")

    # Real-time collaboration status
    st.markdown("### 🟢 Collaboration Status")
    if st.session_state.collaborators:
        st.success(f"👥 {len(st.session_state.collaborators)} collaborators")
        for collaborator in st.session_state.collaborators[:3]:  # Show first 3
            st.caption(f"• {collaborator}")
        if len(st.session_state.collaborators) > 3:
            st.caption(f"... and {len(st.session_state.collaborators) - 3} more")
    else:
        st.info("ℹ️ Add collaborators to enable real-time collaboration")

    # Activity feed
    st.markdown("### 📝 Recent Activity")
    comments = get_comments()
    if comments:
        for comment in comments[-3:]:  # Show last 3 comments
            st.caption(f"💬 {comment['author']}: {comment['text'][:50]}{'...' if len(comment['text']) > 50 else ''}")
            st.caption(f"🕒 {comment['timestamp'][:16]}")
    else:
        st.caption("No recent activity")

    # Notifications
    st.markdown("### 🔔 Notifications")
    if st.session_state.get("adversarial_running"):
        st.info("🔄 Adversarial testing in progress")
    if st.session_state.get("evolution_running"):
        st.info("🔄 Evolution process in progress")
    if st.session_state.protocol_versions and len(st.session_state.protocol_versions) > len(
            st.session_state.get("notified_versions", [])):
        st.success("✅ New version saved")
        # Update notified versions
        st.session_state.notified_versions = st.session_state.protocol_versions.copy()

    # Template Marketplace
    st.markdown("---")
    st.subheader("🛍️ Template Marketplace")

    # Search and browse templates
    search_query = st.text_input("Search Templates", key="template_search")

    # Category filter
    categories = list_template_categories()
    selected_category = st.selectbox("Category", ["All"] + categories, key="template_category")

    # Display templates
    if search_query:
        templates = search_templates(search_query)
        st.write(f"### Search Results ({len(templates)} found)")
    elif selected_category != "All":
        template_names = list_templates_in_category(selected_category)
        templates = [(selected_category, name, get_template_details(selected_category, name)) for name in
                     template_names]
        st.write(f"### {selected_category} Templates ({len(templates)} found)")
    else:
        # Show popular templates
        templates = get_popular_templates(5)
        st.write("### Popular Templates")

    # Display template cards
    for category, template_name, details in templates[:10]:  # Limit to first 10
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{template_name}**")
            st.caption(details.get("description", "")[:100] + "..." if len(
                details.get("description", "")) > 100 else details.get("description", ""))
        with col2:
            st.caption(f"⭐ {details.get('rating', 0)}")
            st.caption(f"📥 {details.get('downloads', 0):,}")
        with col3:
            if st.button("📥 Load", key=f"load_template_{category}_{template_name}"):
                # In a real implementation, this would load the template
                st.success(f"Loaded template: {template_name}")
                st.rerun()

    # External Integrations
    st.markdown("---")
    st.subheader("🔌 External Integrations")

    # Show integration status
    integrations = list_external_integrations()
    for integration_name in integrations:
        is_authenticated = st.session_state.get(f"{integration_name}_authenticated", False)
        integration = get_integration_details(integration_name)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{integration['name']}**")
            if is_authenticated:
                st.caption("✅ Connected")
            else:
                st.caption("❌ Not connected")
        with col2:
            if st.button("⚙️" if is_authenticated else "🔗", key=f"config_integration_{integration_name}"):
                # In a real implementation, this would show configuration dialog
                st.info(f"Configure {integration['name']} integration")

    # GitHub Integration
    st.markdown("---")
    st.subheader("🔗 GitHub Integration")

    # GitHub authentication
    if st.session_state.get("github_user"):
        # User is authenticated
        user = st.session_state.github_user
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{user.get('login', 'Unknown')}**")
            st.caption("✅ Connected to GitHub")
        with col2:
            if st.button("Disconnect", key="disconnect_github", use_container_width=True):
                # Disconnect from GitHub
                if "github_user" in st.session_state:
                    del st.session_state.github_user
                if "github_token" in st.session_state:
                    del st.session_state.github_token
                st.success("Disconnected from GitHub")
                st.rerun()

        # Repository linking
        st.markdown("### 📂 Repositories")
        linked_repos = list_linked_github_repositories()
        if linked_repos:
            for repo_name in linked_repos:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.caption(repo_name)
                with col2:
                    if st.button("SetBranch", key=f"set_branch_{repo_name}"):
                        # In a real implementation, this would set the branch
                        st.info(f"Set branch for {repo_name}")
                with col3:
                    if st.button("❌", key=f"unlink_repo_{repo_name}"):
                        if unlink_github_repository(repo_name):
                            st.success(f"Unlinked {repo_name}")
                            st.rerun()
        else:
            st.info("No repositories linked yet")

        # Link new repository
        st.markdown("### 🔗 Link New Repository")
        repo_name = st.text_input("Repository Name", placeholder="owner/repository")
        if repo_name and st.button("Link Repository"):
            token = st.session_state.get("github_token")
            if token:
                if link_github_repository(token, repo_name):
                    st.success(f"Linked repository: {repo_name}")
                    st.rerun()
            else:
                st.error("GitHub token not found")
    else:
        # User is not authenticated
        github_token = st.text_input("GitHub Personal Access Token", type="password", key="github_token_input")
        if github_token and st.button("Authenticate with GitHub", key="auth_github"):
            if authenticate_github(github_token):
                # Store the token
                st.session_state.github_token = github_token
                st.success("Successfully authenticated with GitHub!")
                st.rerun()

        st.caption("Generate a token with 'repo' permissions at [GitHub Settings](https://github.com/settings/tokens)")

    # AI-Powered Features
    st.markdown("---")
    st.subheader("🤖 AI-Powered Features")

    # ML Model Suggestions
    st.markdown("### 🧠 Protocol Enhancement")
    if st.button("Get AI Suggestions"):
        # In a real implementation, this would analyze the current protocol
        st.info("Analyzing protocol with AI models...")

    # Report Generation
    st.markdown("### 📋 Report Generation")
    report_types = list_report_templates()
    selected_report = st.selectbox("Report Type", report_types, key="report_type")
    if selected_report and st.button("Generate Report"):
        # In a real implementation, this would generate a report
        st.success(f"Generated {selected_report} report")

tab1, tab2, tab3 = st.tabs(["🔄 Evolution", "⚔️ Adversarial Testing", "🐙 GitHub"])

with tab1:
    # Content input section
    st.subheader("📝 Content Input")
    st.text_area("Paste your draft content here:", height=300, key="protocol_text",
                 disabled=st.session_state.adversarial_running)

    # Content Templates
    templates = list_protocol_templates()
    if templates:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_template = st.selectbox("Load Template", [""] + templates, key="load_template_select")
        with col2:
            if selected_template and st.button("Load Selected Template", key="load_template_btn",
                                               use_container_width=True):
                template_content = load_protocol_template(selected_template)
                st.session_state.protocol_text = template_content
                st.success(f"Loaded template: {selected_template}")
                st.rerun()

    # AI Recommendations
    if st.session_state.protocol_text.strip():
        with st.expander("🤖 AI Recommendations", expanded=False):
            recommendations = generate_protocol_recommendations(st.session_state.protocol_text)
            suggested_template = suggest_protocol_template(st.session_state.protocol_text)

            st.markdown("### 💡 Improvement Suggestions")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

            st.markdown(f"### 📋 Suggested Template: **{suggested_template}**")
            if st.button("Load Suggested Template"):
                template_content = load_protocol_template(suggested_template)
                if template_content:
                    st.session_state.protocol_text = template_content
                    st.success(f"Loaded template: {suggested_template}")
                    st.rerun()

    # Action buttons
    st.markdown("---")
    c1, c2 = st.columns(2)
    run_button = c1.button("🚀 Start Evolution", type="primary", disabled=st.session_state.evolution_running,
                           use_container_width=True)
    stop_button = c2.button("⏹️ Stop Evolution", disabled=not st.session_state.evolution_running,
                            use_container_width=True)

    # Results section
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("📄 Current Best Content")
        proto_out = st.empty()

        # Content Analysis
        if st.session_state.evolution_current_best or st.session_state.protocol_text:
            current_content = st.session_state.evolution_current_best or st.session_state.protocol_text
            with st.expander("🔍 Content Analysis", expanded=False):
                complexity = calculate_protocol_complexity(current_content)
                structure = extract_protocol_structure(current_content)

                # Use the new CSS class for the analysis card
                st.markdown('<div class="protocol-analysis-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Content Metrics")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("_WORDS", complexity["word_count"])
                    st.metric("_SENTENCES", complexity["sentence_count"])
                    st.metric("_COMPLEXITY", complexity["complexity_score"])

                with col2:
                    st.metric("PARAGRAPHS", complexity["paragraph_count"])
                    st.metric("UNIQUE WORDS", complexity["unique_words"])
                    st.metric("SECTIONS", structure["section_count"])
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### 🧩 Structure Analysis")
                col3, col4 = st.columns(2)
                with col3:
                    st.write("Numbered Steps:", "✅" if structure["has_numbered_steps"] else "❌")
                    st.write("Bullet Points:", "✅" if structure["has_bullet_points"] else "❌")
                    st.write("Headers:", "✅" if structure["has_headers"] else "❌")
                with col4:
                    st.write("Preconditions:", "✅" if structure["has_preconditions"] else "❌")
                    st.write("Postconditions:", "✅" if structure["has_postconditions"] else "❌")
                    st.write("Error Handling:", "✅" if structure["has_error_handling"] else "❌")
    with right:
        st.subheader("🔍 Logs")
        log_out = st.empty()

        # Comments section
        with st.expander("💬 Comments", expanded=False):
            comments = get_comments()
            if comments:
                for comment in comments:
                    st.markdown(f"**{comment['author']}** ({comment['timestamp'][:16]})")
                    st.markdown(f"> {comment['text']}")
                    st.markdown("---")

            new_comment = st.text_area("Add a comment", key="new_comment")
            if st.button("Post Comment"):
                if new_comment.strip():
                    add_comment(new_comment)
                    st.success("Comment added!")
                    st.rerun()

    # Display the current state from the session state
    with st.session_state.thread_lock:
        current_log = "\n".join(st.session_state.evolution_log)
        current_content = st.session_state.evolution_current_best or st.session_state.protocol_text

    log_out.code(current_log, language="text")
    proto_out.code(current_content, language="markdown")

    # Enhanced visualization for evolution process
    if st.session_state.evolution_running or st.session_state.evolution_current_best:
        st.markdown("---")
        st.subheader("📊 Evolution Progress")

        # Progress metrics
        if st.session_state.evolution_current_best:
            original_complexity = calculate_protocol_complexity(st.session_state.protocol_text)
            current_complexity = calculate_protocol_complexity(st.session_state.evolution_current_best)

            progress_col1, progress_col2, progress_col3 = st.columns(3)
            progress_col1.metric("📝 Original Length", f"{original_complexity['word_count']} words")
            progress_col2.metric("📝 Current Length", f"{current_complexity['word_count']} words")
            progress_col3.metric("📈 Improvement",
                                 f"{current_complexity['word_count'] - original_complexity['word_count']} words",
                                 f"{((current_complexity['word_count'] / max(1, original_complexity['word_count'])) - 1) * 100:.1f}%")

    # If evolution is running, sleep for 1 second and then rerun to update the UI
    if st.session_state.evolution_running:
        time.sleep(1)
        st.rerun()


def render_adversarial_testing_tab():
    st.header("🔴🔵 Adversarial Testing with Multi-LLM Consensus")

    # Add a brief introduction
    st.markdown("""
    > **How it works:** Adversarial Testing uses two teams of AI models to improve your content:
    > - **🔴 Red Team** finds flaws and vulnerabilities
    > - **🔵 Blue Team** fixes the identified issues
    > The process repeats until your content reaches the desired confidence level.
    """)

    # Project Information Section
    st.markdown("---")
    st.subheader("📁 Project Information")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("Project Name", key="project_name")
        st.text_area("Project Description", key="project_description", height=100)
    with col2:
        # Version control
        if st.button("💾 Save Version"):
            if st.session_state.protocol_text.strip():
                version_name = st.text_input("Version Name", f"Version {len(st.session_state.protocol_versions) + 1}")
                comment = st.text_area("Comment", height=100, key="version_comment")
                if st.button("✅ Confirm Save"):
                    version_id = create_new_version(st.session_state.protocol_text, version_name, comment)
                    st.success(f"✅ Version saved! ID: {version_id[:8]}")
                    st.rerun()

        # Show version history
        versions = get_version_history()
        if versions:
            st.markdown("### 📚 Versions")
            for version in reversed(versions[-5:]):  # Show last 5 versions
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"{version['name']} ({version['timestamp'][:10]})",
                                 key=f"load_version_{version['id']}"):
                        load_version(version['id'])
                        st.success(f"✅ Loaded version: {version['name']}")
                        st.rerun()
                with col2:
                    st.caption(f"v{version['id'][:8]}")

    # Collaborative Features
    with st.expander("👥 Collaborative Features", expanded=False):
        st.markdown("### 🤝 Team Collaboration")
        collaborators = st.multiselect("Add Collaborators (email addresses)",
                                       st.session_state.collaborators,
                                       key="collaborators")

        st.markdown("### 💬 Comments & Discussions")
        comments = get_comments()
        if comments:
            for comment in comments:
                st.markdown(f"**{comment['author']}** ({comment['timestamp'][:16]})")
                st.markdown(f"> {comment['text']}")
                st.markdown("---")

        new_comment = st.text_area("Add a comment", key="new_comment")
        if st.button("📤 Post Comment"):
            if new_comment.strip():
                add_comment(new_comment)
                st.success("✅ Comment added!")
                st.rerun()

        st.markdown("### 🏷️ Tags")
        tags = st.multiselect("Add tags to organize this project",
                              st.session_state.tags,
                              key="tags")

    # Quick Start Wizard
    with st.expander("⚡ Quick Start Wizard", expanded=True):
        st.markdown("### 🚀 Get Started in 3 Easy Steps")

        # Step 1: Configure API Key
        st.markdown("#### 1️⃣ Configure OpenRouter API Key")
        openrouter_key = st.text_input("🔑 Enter your OpenRouter API Key", type="password", key="wizard_openrouter_key")
        if openrouter_key:
            st.session_state.openrouter_key = openrouter_key
            st.success("✅ API key saved!")
        else:
            st.info("ℹ️ Need an API key? Get one at [OpenRouter.ai](https://openrouter.ai/keys)")

        # Step 2: Select Models
        st.markdown("#### 2️⃣ Select AI Models")
        if openrouter_key:
            models = get_openrouter_models(openrouter_key)
            if models:
                model_names = [m['id'] for m in models if isinstance(m, dict) and 'id' in m][:10]  # Top 10 models

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔴 Red Team (Critics)**")
                    red_models = st.multiselect(
                        "Select 2-3 models for finding flaws",
                        options=model_names,
                        default=model_names[:2] if len(model_names) >= 2 else model_names,
                        key="wizard_red_models"
                    )
                    st.session_state.red_team_models = red_models

                with col2:
                    st.markdown("**🔵 Blue Team (Fixers)**")
                    blue_models = st.multiselect(
                        "Select 2-3 models for fixing issues",
                        options=model_names,
                        default=model_names[2:4] if len(model_names) >= 4 else model_names[-2:],
                        key="wizard_blue_models"
                    )
                    st.session_state.blue_team_models = blue_models

                if red_models and blue_models:
                    st.success(f"✅ {len(red_models)} red team and {len(blue_models)} blue team models selected!")
                else:
                    st.info("ℹ️ Please select at least one model for each team")
            else:
                st.warning("⚠️ Unable to fetch models. Please check your API key.")
        else:
            st.info("ℹ️ Please enter your OpenRouter API key to select models")

        # Step 3: Load or Create Content
        st.markdown("#### 3️⃣ Load or Create Your Content")
        templates = list_protocol_templates()
        if templates:
            selected_template = st.selectbox("📝 Choose a template or start from scratch", [""] + templates,
                                             key="wizard_template")
            if selected_template:
                template_content = load_protocol_template(selected_template)
                st.session_state.protocol_text = template_content
                st.success(f"✅ Loaded {selected_template} template!")

        # Quick content editor
        content = st.text_area("✏️ Or paste/write your content here",
                               value=st.session_state.protocol_text,
                               height=200,
                               key="wizard_content_text")
        if content != st.session_state.protocol_text:
            st.session_state.protocol_text = content

        # Quick start button
        if st.button("🚀 Quick Start Adversarial Testing",
                     disabled=not (
                             openrouter_key and st.session_state.red_team_models and st.session_state.blue_team_models and content.strip()),
                     type="primary",
                     use_container_width=True):
            st.success(
                "🎉 Ready to go! Scroll down to configure advanced settings or click 'Start Adversarial Testing' below.")
            st.rerun()

    # Project controls
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        if st.button("💾 Save Project"):
            project_data = export_project()
            st.download_button(
                label="📥 Download Project",
                data=json.dumps(project_data, indent=2),
                file_name=f"{st.session_state.project_name.replace(' ', '_')}_project.json",
                mime="application/json",
                use_container_width=True,
            )
    with col2:
        uploaded_file = st.file_uploader("📁 Import Project", type=["json"])
        if uploaded_file:
            try:
                project_data = json.load(uploaded_file)
                if import_project(project_data):
                    st.success("Project imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing project: {e}")
    with col3:
        if st.button("📋 Export Report"):
            if st.session_state.adversarial_results:
                # Create tabs for different export formats
                export_format = st.radio("Export Format", ["HTML", "PDF", "DOCX"], horizontal=True)

                if export_format == "HTML":
                    html_content = generate_html_report(st.session_state.adversarial_results)
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_content,
                        file_name="adversarial_testing_report.html",
                        mime="text/html"
                    )
                elif export_format == "PDF" and HAS_FPDF:
                    pdf_content = generate_pdf_report(st.session_state.adversarial_results)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_content,
                        file_name="adversarial_testing_report.pdf",
                        mime="application/pdf"
                    )
                elif export_format == "DOCX" and HAS_DOCX:
                    docx_content = generate_docx_report(st.session_state.adversarial_results)
                    st.download_button(
                        label="📥 Download DOCX Report",
                        data=docx_content,
                        file_name="adversarial_testing_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                elif export_format == "PDF" and not HAS_FPDF:
                    st.error(
                        "FPDF library not installed. Please install it with 'pip install fpdf' to export PDF reports.")
                elif export_format == "DOCX" and not HAS_DOCX:
                    st.error(
                        "python-docx library not installed. Please install it with 'pip install python-docx' to export DOCX reports.")
    with col4:
        if st.button("❓ Tutorial"):
            st.session_state.show_adversarial_tutorial = True

    # Sharing and Collaboration Controls
    with st.expander("🔗 Share & Collaborate", expanded=False):
        st.markdown("### 🌐 Public Sharing")
        share_publicly = st.toggle("Share publicly", key="share_publicly")
        if share_publicly:
            st.info("🔒 Your project will be accessible via a public link. Only people with the link can view it.")
            if st.button("🔗 Generate Shareable Link"):
                # In a real implementation, this would generate a real shareable link
                share_link = f"https://open-evolve.app/shared/{uuid.uuid4()}"
                st.code(share_link, language="markdown")
                st.info("📋 Copy this link to share your project with others.")

        st.markdown("### 📧 Invite Collaborators")
        collaborator_emails = st.text_area("Enter email addresses (one per line)",
                                           key="collaborator_emails",
                                           height=100)
        if st.button("✉️ Send Invitations"):
            if collaborator_emails.strip():
                emails = [email.strip() for email in collaborator_emails.split("\n") if email.strip()]
                st.success(f"📧 Sent invitations to {len(emails)} collaborators!")
            else:
                st.warning("📧 Please enter at least one email address.")

        st.markdown("### 📤 Export Options")
        export_options = st.multiselect(
            "Select what to export",
            ["Protocol Versions", "Adversarial Results", "Comments", "Analytics", "Full Project"],
            default=["Full Project"]
        )

        if "Full Project" in export_options:
            st.download_button(
                label="📦 Export Full Project (.json)",
                data=json.dumps(export_project(), indent=2),
                file_name=f"{st.session_state.project_name.replace(' ', '_')}_full_export.json",
                mime="application/json"
            )
        else:
            # Custom export
            if st.button("⚙️ Generate Custom Export"):
                custom_export = {}
                if "Protocol Versions" in export_options:
                    custom_export["versions"] = st.session_state.protocol_versions
                if "Adversarial Results" in export_options:
                    custom_export["results"] = st.session_state.adversarial_results
                if "Comments" in export_options:
                    custom_export["comments"] = st.session_state.comments
                if "Analytics" in export_options:
                    if st.session_state.adversarial_results:
                        custom_export["analytics"] = generate_advanced_analytics(st.session_state.adversarial_results)

                st.download_button(
                    label="📥 Download Custom Export (.json)",
                    data=json.dumps(custom_export, indent=2),
                    file_name=f"{st.session_state.project_name.replace(' ', '_')}_custom_export.json",
                    mime="application/json"
                )

        # Export to different formats
        st.markdown("### 📄 Format Export")
        format_options = st.selectbox("Export Format",
                                      ["Markdown", "PDF", "Word Document", "HTML", "LaTeX", "Plain Text"])
        if st.button(f"🖨️ Export as {format_options}"):
            if format_options == "Markdown":
                st.download_button(
                    label="📥 Download Markdown (.md)",
                    data=st.session_state.protocol_text,
                    file_name=f"{st.session_state.project_name.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
            elif format_options == "PDF":
                st.info("Generating PDF... (This would generate a formatted PDF in a real implementation)")
            elif format_options == "Word Document":
                st.info("Generating Word document... (This would generate a .docx file in a real implementation)")
            elif format_options == "HTML":
                st.download_button(
                    label="📥 Download HTML (.html)",
                    data=f"<html><body><h1>{st.session_state.project_name}</h1><pre>{st.session_state.protocol_text}</pre></body></html>",
                    file_name=f"{st.session_state.project_name.replace(' ', '_')}.html",
                    mime="text/html"
                )
            elif format_options == "LaTeX":
                st.info("Generating LaTeX... (This would generate a .tex file in a real implementation)")
            elif format_options == "Plain Text":
                st.download_button(
                    label="📥 Download Plain Text (.txt)",
                    data=st.session_state.protocol_text,
                    file_name=f"{st.session_state.project_name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

    # Show tutorial modal if requested
    if st.session_state.get("show_adversarial_tutorial", False):
        with st.expander("📘 Adversarial Testing Tutorial", expanded=True):
            st.markdown("""
            ### 🎓 Adversarial Testing Guide

            Adversarial Testing uses two teams of AI models to improve your content:

            #### 🔴 Red Team (Critics)
            - Finds flaws, vulnerabilities, and weaknesses in your content
            - Analyzes for logical gaps, ambiguities, and potential misuse

            #### 🔵 Blue Team (Fixers)
            - Addresses the issues identified by the Red Team
            - Produces improved versions of the content

            #### 🔄 Process
            1. Red Team critiques the content
            2. Blue Team patches the identified issues
            3. Consensus mechanism selects the best patch
            4. Approval check validates the improvements
            5. Process repeats until confidence threshold is reached

            #### ⚙️ Key Parameters
            - **Confidence Threshold**: Percentage of Red Team that must approve the content
            - **Iterations**: Number of improvement cycles to run
            - **Rotation Strategy**: How to select models for each iteration
            - **Custom Mode**: Use your own prompts for testing

            #### 🎯 Tips for Best Results
            - Use diverse models for both teams
            - Set appropriate confidence thresholds (80-95%)
            - Provide clear compliance requirements
            - Use custom prompts for domain-specific testing
            """)

            if st.button("Close Tutorial"):
                st.session_state.show_adversarial_tutorial = False
                st.rerun()

    # Quick Start Guide
    with st.expander("⚡ Quick Start Guide", expanded=False):
        st.markdown("""
        ### 🚀 Getting Started in 3 Steps:

        1. **🔑 Configure OpenRouter**
           - Enter your OpenRouter API key
           - Select models for Red and Blue teams

        2. **📝 Input Your Content**
           - Paste your content or load a template
           - Add compliance requirements if needed

        3. **▶️ Run Adversarial Testing**
           - Adjust parameters as needed
           - Click "Start Adversarial Testing"
           - Monitor progress in real-time
        """)

        if st.button("📋 Load Sample Content"):
            sample_content = """# Sample Security Policy

## Overview
This policy defines security requirements for accessing company systems.

## Scope
Applies to all employees, contractors, and vendors with system access.

## Policy Statements
1. All users must use strong passwords
2. Multi-factor authentication is required for sensitive systems
3. Regular security training is mandatory
4. Incident reporting must occur within 24 hours

## Roles and Responsibilities
- IT Security Team: Enforces policy and monitors compliance
- Employees: Follow security practices and report incidents
- Managers: Ensure team compliance and provide resources

## Compliance
- Audits conducted quarterly
- Violations result in disciplinary action
- Continuous monitoring through SIEM tools

## Exceptions
- Emergency access requests require manager approval
- Temporary exceptions require security team approval

## Review and Updates
- Policy reviewed annually
- Updates approved by CISO"""
            st.session_state.protocol_text = sample_content
            st.success("Sample content loaded! You can now start adversarial testing.")
            st.rerun()

    # OpenRouter Configuration
    st.subheader("🔑 OpenRouter Configuration")
    openrouter_key = st.text_input("OpenRouter API Key", type="password", key="openrouter_key")
    if not openrouter_key:
        st.info("Enter your OpenRouter API key to enable model selection and testing.")
        return

    models = get_openrouter_models(openrouter_key)
    # Update global model metadata with thread safety
    for m in models:
        if isinstance(m, dict) and (mid := m.get("id")):
            with MODEL_META_LOCK:
                MODEL_META_BY_ID[mid] = m
    if not models:
        st.error("No models fetched. Check your OpenRouter key and connection.")
        return

    model_options = sorted([
        f"{m['id']} (Ctx: {m.get('context_length', 'N/A')}, "
        f"In: {_parse_price_per_million(m.get('pricing', {}).get('prompt')) or 'N/A'}/M, "
        f"Out: {_parse_price_per_million(m.get('pricing', {}).get('completion')) or 'N/A'}/M)"
        for m in models if isinstance(m, dict) and "id" in m
    ])

    # Protocol Templates
    st.markdown("---")
    st.subheader("📝 Content Input")

    # Add protocol input guidance
    st.info(
        "💡 **Tip:** Start with a clear, well-structured content. The better your starting point, the better the results.")

    # Protocol editor with enhanced features
    protocol_col1, protocol_col2 = st.columns([3, 1])
    with protocol_col1:
        protocol_text = st.text_area("✏️ Enter or paste your content:",
                                     value=st.session_state.protocol_text,
                                     height=300,
                                     key="protocol_text",
                                     placeholder="Paste your draft content here...\n\nExample:\n# Security Policy\n\n## Overview\nThis policy defines requirements for secure system access.\n\n## Scope\nApplies to all employees and contractors.\n\n## Policy Statements\n1. All users must use strong passwords\n2. Multi-factor authentication is required for sensitive systems\n3. Regular security training is mandatory\n\n## Compliance\nViolations result in disciplinary action.")

    with protocol_col2:
        st.markdown("**📋 Quick Actions**")

        # Template loading
        templates = list_protocol_templates()
        if templates:
            selected_template = st.selectbox("Load Template", [""] + templates, key="adv_load_template_select")
            if selected_template and st.button("📥 Load Template", use_container_width=True):
                template_content = load_protocol_template(selected_template)
                st.session_state.protocol_text = template_content
                st.success(f"Loaded: {selected_template}")
                st.rerun()

        # Sample protocol
        if st.button("🧪 Load Sample", use_container_width=True):
            sample_protocol = """# Sample Security Policy

## Overview
This policy defines security requirements for accessing company systems.

## Scope
Applies to all employees, contractors, and vendors with system access.

## Policy Statements
1. All users must use strong passwords
2. Multi-factor authentication is required for sensitive systems
3. Regular security training is mandatory
4. Incident reporting must occur within 24 hours

## Roles and Responsibilities
- IT Security Team: Enforces policy and monitors compliance
- Employees: Follow security practices and report incidents
- Managers: Ensure team compliance and provide resources

## Compliance
- Audits conducted quarterly
- Violations result in disciplinary action
- Continuous monitoring through SIEM tools

## Exceptions
- Emergency access requests require manager approval
- Temporary exceptions require security team approval

## Review and Updates
- Policy reviewed annually
- Updates approved by CISO"""
            st.session_state.protocol_text = sample_protocol
            st.success("Sample protocol loaded!")
            st.rerun()

        # Clear button
        if st.session_state.protocol_text.strip() and st.button("🗑️ Clear", use_container_width=True):
            st.session_state.protocol_text = ""
            st.rerun()

        # Protocol analysis
        if st.session_state.protocol_text.strip():
            complexity = calculate_protocol_complexity(st.session_state.protocol_text)
            structure = extract_protocol_structure(st.session_state.protocol_text)

            st.markdown("**📊 Quick Stats**")
            st.metric("_WORDS", complexity["word_count"])
            st.metric("_SENTENCES", complexity["sentence_count"])
            st.metric("_COMPLEXITY", complexity["complexity_score"])

            # Structure indicators
            structure_icons = []
            if structure["has_numbered_steps"]:
                structure_icons.append("🔢")
            if structure["has_bullet_points"]:
                structure_icons.append("•")
            if structure["has_headers"]:
                structure_icons.append("#")
            if structure["has_preconditions"]:
                structure_icons.append("🔒")
            if structure["has_postconditions"]:
                structure_icons.append("✅")
            if structure["has_error_handling"]:
                structure_icons.append("⚠️")

            if structure_icons:
                st.markdown(" ".join(structure_icons))

    # AI Recommendations
    if st.session_state.protocol_text.strip():
        with st.expander("🤖 AI Recommendations", expanded=False):
            recommendations = generate_protocol_recommendations(st.session_state.protocol_text)
            suggested_template = suggest_protocol_template(st.session_state.protocol_text)

            st.markdown("### 💡 Improvement Suggestions")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

            st.markdown(f"### 📋 Suggested Template: **{suggested_template}**")
            if st.button("Load Suggested Template", key="adv_load_suggested_template"):
                template_content = load_protocol_template(suggested_template)
                if template_content:
                    st.session_state.protocol_text = template_content
                    st.success(f"Loaded template: {suggested_template}")
                    st.rerun()

    # Model Selection
    st.markdown("---")
    st.subheader("🤖 Model Selection")

    # Add model selection guidance
    st.info(
        "💡 **Tip:** Select 3-5 diverse models for each team for best results. Mix small and large models for cost-effectiveness.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 Red Team (Critics)")
        st.caption("Models that find flaws and vulnerabilities in your protocol")

        # Quick selection buttons
        if model_options:
            quick_red_models = st.multiselect(
                "Quick Select Red Team Models",
                options=[opt.split(" (")[0] for opt in model_options[:10]],  # First 10 models
                default=st.session_state.red_team_models[:3] if st.session_state.red_team_models else [],
                key="quick_red_select"
            )
            if quick_red_models:
                st.session_state.red_team_models = quick_red_models

        if HAS_STREAMLIT_TAGS:
            red_team_selected_full = st_tags(
                label="Search and select models:",
                text="Type to search models...",
                value=st.session_state.red_team_models,
                suggestions=model_options,
                key="red_team_select"
            )
            # Robust model ID extraction from descriptive string
            red_team_models = []
            for m in red_team_selected_full:
                # Extract model ID by splitting on first occurrence of " (" or using entire string
                # if not found
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    red_team_models.append(model_id)
            st.session_state.red_team_models = sorted(list(set(red_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            red_team_input = st.text_input("Enter Red Team models (comma-separated):",
                                           value=",".join(st.session_state.red_team_models))
            st.session_state.red_team_models = sorted(
                list(set([model.strip() for model in red_team_input.split(",") if model.strip()])))

        # Model count indicator
        st.caption(f"Selected: {len(st.session_state.red_team_models)} models")

    with col2:
        st.markdown("#### 🔵 Blue Team (Fixers)")
        st.caption("Models that patch the identified flaws and improve the protocol")

        # Quick selection buttons
        if model_options:
            quick_blue_models = st.multiselect(
                "Quick Select Blue Team Models",
                options=[opt.split(" (")[0] for opt in model_options[:10]],  # First 10 models
                default=st.session_state.blue_team_models[:3] if st.session_state.blue_team_models else [],
                key="quick_blue_select"
            )
            if quick_blue_models:
                st.session_state.blue_team_models = quick_blue_models

        if HAS_STREAMLIT_TAGS:
            blue_team_selected_full = st_tags(
                label="Search and select models:",
                text="Type to search models...",
                value=st.session_state.blue_team_models,
                suggestions=model_options,
                key="blue_team_select"
            )
            # Robust model ID extraction from descriptive string
            blue_team_models = []
            for m in blue_team_selected_full:
                # Extract model ID by splitting on first occurrence of " (" or using entire string
                # if not found
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    blue_team_models.append(model_id)
            st.session_state.blue_team_models = sorted(list(set(blue_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            blue_team_input = st.text_input("Enter Blue Team models (comma-separated):",
                                            value=",".join(st.session_state.blue_team_models))
            st.session_state.blue_team_models = sorted(
                list(set([model.strip() for model in blue_team_input.split(",") if model.strip()])))

        # Model count indicator
        st.caption(f"Selected: {len(st.session_state.blue_team_models)} models")

    # Model selection validation
    if st.session_state.red_team_models and st.session_state.blue_team_models:
        total_models = len(st.session_state.red_team_models) + len(st.session_state.blue_team_models)
        if total_models > 10:
            st.warning(
                f"⚠️ You have selected {total_models} models. Consider reducing the number to control costs and processing time.")
        else:
            st.success(
                f"✅ Ready! {len(st.session_state.red_team_models)} red team and {len(st.session_state.blue_team_models)} blue team models selected.")
    elif not st.session_state.red_team_models or not st.session_state.blue_team_models:
        st.info("ℹ️ Please select at least one model for each team to proceed.")

    # Testing Parameters
    st.markdown("---")
    st.subheader("🧪 Testing Parameters")

    # Preset Selector
    with st.expander("🎯 Presets", expanded=True):
        st.markdown("### 🚀 Quick Start with Presets")
        st.info("💡 **Tip:** Use presets to quickly configure adversarial testing for common scenarios.")

        preset_names = list_adversarial_presets()
        if preset_names:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_preset = st.selectbox("Choose a preset configuration", [""] + preset_names,
                                               key="preset_selector")
            with col2:
                if st.button("Apply Preset", key="apply_preset_btn", use_container_width=True):
                    if selected_preset and apply_adversarial_preset(selected_preset):
                        st.success(f"✅ Applied {selected_preset} preset!")
                        st.rerun()
                    elif selected_preset:
                        st.error("❌ Failed to apply preset.")

            # Show preset details
            if selected_preset:
                preset = load_adversarial_preset(selected_preset)
                if preset:
                    st.markdown(f"**{preset['name']}**")
                    st.caption(preset['description'])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**🔴 Red Team Models:**")
                        for model in preset.get("red_team_models", []):
                            st.code(model, language="markdown")
                    with col2:
                        st.write("**🔵 Blue Team Models:**")
                        for model in preset.get("blue_team_models", []):
                            st.code(model, language="markdown")
                    st.write("**⚙️ Settings:**")
                    st.write(f"- Iterations: {preset.get('min_iter', 3)}-{preset.get('max_iter', 10)}")
                    st.write(f"- Confidence Threshold: {preset.get('confidence_threshold', 85)}%")
                    st.write(f"- Review Type: {preset.get('review_type', 'General SOP')}")

        # Advanced Testing Strategies
        st.markdown("### 🧠 Advanced Testing Strategies")
        strategy_options = st.multiselect(
            "Select testing strategies to enable:",
            ["Adaptive Testing", "Category-Focused Testing", "Performance-Based Rotation", "Continuous Learning"],
            default=[],
            key="advanced_testing_strategies"
        )

        if "Adaptive Testing" in strategy_options:
            st.info("🔄 **Adaptive Testing**: Automatically adjusts testing intensity based on results.")

        if "Category-Focused Testing" in strategy_options:
            focus_category = st.selectbox(
                "Focus on specific issue category:",
                ["", "Security", "Compliance", "Clarity", "Completeness", "Efficiency"],
                key="category_focus"
            )
            if focus_category:
                st.info(f"🎯 **Category Focus**: Testing will emphasize {focus_category.lower()} issues.")

        if "Performance-Based Rotation" in strategy_options:
            st.info("⚡ **Performance-Based Rotation**: Automatically rotates models based on performance metrics.")

        if "Continuous Learning" in strategy_options:
            st.info("📚 **Continuous Learning**: Uses historical results to improve future testing runs.")

    # Custom Mode Toggle
    use_custom_mode = st.toggle("🔧 Use Custom Mode", key="adversarial_custom_mode",
                                help="Enable custom prompts and configurations for adversarial testing")

    if use_custom_mode:
        with st.expander("🔧 Custom Prompts", expanded=True):
            st.text_area("Red Team Prompt (Critique)",
                         value=RED_TEAM_CRITIQUE_PROMPT,
                         key="adversarial_custom_red_prompt",
                         height=200,
                         help="Custom prompt for the red team to find flaws in the protocol")

            st.text_area("Blue Team Prompt (Patch)",
                         value=BLUE_TEAM_PATCH_PROMPT,
                         key="adversarial_custom_blue_prompt",
                         height=200,
                         help="Custom prompt for the blue team to patch the identified flaws")

            st.text_area("Approval Prompt",
                         value=APPROVAL_PROMPT,
                         key="adversarial_custom_approval_prompt",
                         height=150,
                         help="Custom prompt for final approval checking")

    # Review Type Selection
    review_types = ["Auto-Detect", "General SOP", "Code Review", "Plan Review"]
    st.selectbox("Review Type", review_types, key="adversarial_review_type",
                 help="Select the type of review to perform. Auto-Detect will analyze the content and choose the appropriate review type.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Min iterations", 1, 50, key="adversarial_min_iter")
        st.number_input("Max iterations", 1, 200, key="adversarial_max_iter")
    with c2:
        st.slider("Confidence threshold (%)", 50, 100, key="adversarial_confidence",
                  help="Stop if this % of Red Team approves the SOP.")
    with c3:
        st.number_input("Max tokens per model", 1000, 100000, key="adversarial_max_tokens")
        st.number_input("Max parallel workers", 1, 24, key="adversarial_max_workers")
    with c4:
        st.toggle("Force JSON mode", key="adversarial_force_json",
                  help="Use model's built-in JSON mode if available. Increases reliability.")
        st.text_input("Deterministic seed", key="adversarial_seed", help="Integer for reproducible runs.")
        st.selectbox("Rotation Strategy",
                     ["None", "Round Robin", "Random Sampling", "Performance-Based", "Staged", "Adaptive",
                      "Focus-Category"], key="adversarial_rotation_strategy")
        if st.session_state.adversarial_rotation_strategy == "Staged":
            st.text_area("Staged Rotation Config (JSON)", key="adversarial_staged_rotation_config", height=150, help="""
[{"red": ["model1", "model2"], "blue": ["model3"]},
 {"red": ["model4"], "blue": ["model5", "model6"]}]
""")
        st.number_input("Red Team Sample Size", 1, 100, key="adversarial_red_team_sample_size")
        st.number_input("Blue Team Sample Size", 1, 100, key="adversarial_blue_team_sample_size")

    st.text_area("Compliance Requirements", key="compliance_requirements", height=150,
                 help="Enter any compliance requirements that the red team should check for.")

    # Advanced customization options
    with st.expander("⚙️ Advanced Customization", expanded=False):
        st.markdown("### 🎯 Target Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Target Complexity Score", 0, 100, key="adversarial_target_complexity",
                            help="Target complexity score for the final protocol (0-100)")
        with col2:
            st.number_input("Target Length (words)", 0, 10000, key="adversarial_target_length",
                            help="Target length for the final protocol in words (0 = no limit)")

        st.markdown("### 🧠 Intelligence Settings")
        st.slider("Critique Depth", 1, 10, key="adversarial_critique_depth",
                  help="How deeply the red team should analyze the protocol (1-10)")
        st.slider("Patch Quality", 1, 10, key="adversarial_patch_quality",
                  help="Quality level for blue team patches (1-10)")

        st.markdown("### 📊 Evaluation Settings")
        st.toggle("Detailed Issue Tracking", key="adversarial_detailed_tracking",
                  help="Track issues by category and severity in detail")
        st.toggle("Performance Analytics", key="adversarial_performance_analytics",
                  help="Show detailed model performance analytics")

        st.markdown("### 🔄 Iteration Controls")
        st.toggle("Early Stopping", key="adversarial_early_stopping",
                  help="Stop early if no improvement is detected")
        st.number_input("Early Stopping Patience", 1, 10, key="adversarial_early_stopping_patience",
                        help="Number of iterations to wait before early stopping")

        st.markdown("### 🎨 Style Customization")
        st.selectbox("Writing Style", ["Professional", "Concise", "Detailed", "Casual", "Technical", "Executive"],
                     key="adversarial_writing_style",
                     help="Preferred writing style for the final protocol")
        st.selectbox("Tone", ["Neutral", "Authoritative", "Friendly", "Strict", "Persuasive"],
                     key="adversarial_tone",
                     help="Desired tone for the protocol")
        st.text_input("Custom Style Instructions",
                      key="adversarial_custom_style",
                      help="Additional style instructions for the protocol writer")

        st.markdown("### 🛡️ Security Settings")
        st.toggle("Include Security Headers", key="adversarial_include_security_headers",
                  help="Add security-focused headers to the protocol")
        st.toggle("Include Compliance Checks", key="adversarial_include_compliance_checks",
                  help="Automatically add compliance-related sections")
        st.text_area("Custom Security Requirements",
                     key="adversarial_custom_security",
                     height=100,
                     help="Additional security requirements to enforce")

        st.markdown("### 📦 Format Options")
        st.selectbox("Output Format", ["Markdown", "Plain Text", "HTML", "LaTeX"],
                     key="adversarial_output_format",
                     help="Desired output format for the final protocol")
        st.toggle("Include Table of Contents", key="adversarial_include_toc",
                  help="Add automatically generated table of contents")
        st.toggle("Include Revision History", key="adversarial_include_revision_history",
                  help="Track changes with revision history section")

        st.markdown("### 🧪 Experimental Features")
        st.toggle("Use Chain-of-Thought Reasoning", key="adversarial_use_cot",
                  help="Enable chain-of-thought reasoning for deeper analysis")
        st.toggle("Include Confidence Intervals", key="adversarial_include_confidence",
                  help="Add confidence intervals to issue severity ratings")
        st.toggle("Enable Self-Critique", key="adversarial_enable_self_critique",
                  help="Have models critique their own suggestions before finalizing")

        st.markdown("### ⚡ Performance Optimization")
        st.toggle("Auto-Optimize Model Selection", key="adversarial_auto_optimize_models",
                  help="Automatically select optimal models based on protocol complexity and budget")
        budget_limit = st.number_input("Budget Limit (USD)", 0.0, 100.0, 0.0, 0.1,
                                       key="adversarial_budget_limit",
                                       help="Maximum budget for this testing session (0 = no limit)")

        # Performance suggestions button
        if st.button("💡 Get Performance Suggestions"):
            current_config = {
                "red_team_models": st.session_state.red_team_models,
                "blue_team_models": st.session_state.blue_team_models,
                "adversarial_max_iter": st.session_state.adversarial_max_iter,
                "protocol_text": st.session_state.protocol_text
            }
            suggestions = suggest_performance_improvements(current_config)
            st.markdown("### 🚀 Performance Suggestions")
            for suggestion in suggestions:
                st.write(suggestion)

        # Time and cost estimation
        if st.button("⏱️ Estimate Time & Cost"):
            protocol_length = len(st.session_state.protocol_text.split())
            estimate = estimate_testing_time_and_cost(
                st.session_state.red_team_models,
                st.session_state.blue_team_models,
                st.session_state.adversarial_max_iter,
                protocol_length
            )
            st.markdown("### 📊 Time & Cost Estimate")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏰ Est. Time", f"{estimate['estimated_time_minutes']} min")
            col2.metric("💰 Est. Cost", f"${estimate['estimated_cost_usd']:.4f}")
            col3.metric("🔄 Operations", f"{estimate['total_operations']:,}")
            col4.metric("🔤 Tokens", f"{estimate['total_tokens_estimated']:,}")

    all_models = sorted(list(set(st.session_state.red_team_models + st.session_state.blue_team_models)))
    if all_models:
        with st.expander("🔧 Per-Model Configuration", expanded=False):
            for model_id in all_models:
                st.markdown(f"**{model_id}**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.slider(f"Temp##{model_id}", 0.0, 2.0, 0.7, 0.1, key=f"temp_{model_id}")
                cc2.slider(f"Top-P##{model_id}", 0.0, 1.0, 1.0, 0.1, key=f"topp_{model_id}")
                cc3.slider(f"Freq Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"freqpen_{model_id}")
                cc4.slider(f"Pres Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"prespen_{model_id}")

    # Metrics Dashboard Preview
    with st.expander("📊 Metrics Dashboard Preview", expanded=False):
        st.markdown("### 📊 Real-time Metrics (During Testing)")
        st.info("These metrics will be updated in real-time during adversarial testing:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Current Confidence", "0.0%")
            st.metric("💰 Est. Cost (USD)", "$0.0000")
            st.metric("🔤 Prompt Tokens", "0")
        with col2:
            st.metric("🔄 Iterations", "0/0")
            st.metric("📝 Completion Tokens", "0")
            st.metric("⚡ Avg Response Time", "0ms")

        st.markdown("### 📈 Confidence Trend")
        st.line_chart([0, 0, 0, 0, 0])  # Placeholder chart

        st.markdown("### 🏆 Top Performing Models")
        st.write("Model performance rankings will appear here during testing.")

    # Start/Stop buttons for adversarial testing
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    start_button = col1.button("🚀 Start Adversarial Testing", type="primary",
                               disabled=st.session_state.adversarial_running or not st.session_state.protocol_text.strip(),
                               use_container_width=True)
    stop_button = col2.button("⏹️ Stop Adversarial Testing",
                              disabled=not st.session_state.adversarial_running,
                              use_container_width=True)

    # Progress and status section
    if st.session_state.adversarial_running or st.session_state.adversarial_status_message:
        status_container = st.container()
        with status_container:
            # Enhanced status display
            if st.session_state.adversarial_status_message:
                # Use different colors based on status message content
                status_msg = st.session_state.adversarial_status_message
                if "Success" in status_msg or "✅" in status_msg:
                    st.success(status_msg)
                elif "Error" in status_msg or "💥" in status_msg or "⚠️" in status_msg:
                    st.error(status_msg)
                elif "Stop" in status_msg or "⏹️" in status_msg:
                    st.warning(status_msg)
                else:
                    st.info(status_msg)

            # Enhanced progress tracking
            if st.session_state.adversarial_running:
                # Progress bar with iteration info
                current_iter = len(st.session_state.get("adversarial_confidence_history", []))
                max_iter = st.session_state.adversarial_max_iter
                progress = min(current_iter / max(1, max_iter), 1.0)

                # Progress bar with percentage
                st.progress(progress, text=f"Iteration {current_iter}/{max_iter} ({int(progress * 100)}%)")

                # Real-time metrics
                if st.session_state.get("adversarial_confidence_history"):
                    current_confidence = st.session_state.adversarial_confidence_history[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📊 Current Confidence", f"{current_confidence:.1f}%")
                    col2.metric("💰 Est. Cost (USD)", f"${st.session_state.adversarial_cost_estimate_usd:.4f}")
                    col3.metric("🔤 Prompt Tokens", f"{st.session_state.adversarial_total_tokens_prompt:,}")
                    col4.metric("📝 Completion Tokens", f"{st.session_state.adversarial_total_tokens_completion:,}")

                # Enhanced logs with auto-scroll
                with st.expander("🔍 Real-time Logs", expanded=True):
                    if st.session_state.adversarial_log:
                        # Show last 50 entries instead of 20 for better visibility
                        log_content = "\n".join(st.session_state.adversarial_log[-50:])
                        st.text_area("Activity Log", value=log_content, height=300,
                                     key="adversarial_log_display",
                                     help="Auto-updating log of adversarial testing activities")
                    else:
                        st.info("⏳ Waiting for adversarial testing to start...")

            # If adversarial testing has results, show them with enhanced visualization
            if st.session_state.adversarial_results and not st.session_state.adversarial_running:
                with st.expander("🏆 Adversarial Testing Results", expanded=True):
                    results = st.session_state.adversarial_results

                    # Enhanced metrics dashboard with better organization
                    st.markdown("### 📊 Performance Summary")

                    # Main metrics in cards with enhanced styling
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("✅ Final Approval Rate", f"{results.get('final_approval_rate', 0):.1f}%")
                    col2.metric("🔄 Iterations Completed", len(results.get('iterations', [])))
                    col3.metric("💰 Total Cost (USD)", f"${results.get('cost_estimate_usd', 0):.4f}")
                    col4.metric("🔤 Total Tokens",
                                f"{results.get('tokens', {}).get('prompt', 0) + results.get('tokens', {}).get('completion', 0):,}")

                    # Add a visual summary dashboard
                    st.markdown("### 📊 Visual Summary")
                    # Create a summary visualization
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">Approval Rate Progress</div>', unsafe_allow_html=True)
                        if results.get('iterations'):
                            confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0)
                                                  for iter in results.get('iterations', [])]
                            st.line_chart(confidence_history)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with summary_col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">Issue Severity Distribution</div>',
                                    unsafe_allow_html=True)
                        if results.get('iterations'):
                            # Aggregate severity data
                            severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                            for iteration in results.get('iterations', []):
                                critiques = iteration.get("critiques", [])
                                for critique in critiques:
                                    critique_json = critique.get("critique_json", {})
                                    issues = critique_json.get("issues", [])
                                    for issue in issues:
                                        severity = issue.get("severity", "low").lower()
                                        if severity in severity_counts:
                                            severity_counts[severity] += 1

                            # Create a simple bar chart representation
                            max_count = max(severity_counts.values()) if severity_counts.values() else 1
                            for severity, count in severity_counts.items():
                                if count > 0:
                                    bar_length = int((count / max_count) * 20)
                                    emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[severity]
                                    st.write(f"{emoji} {severity.capitalize()}: {'█' * bar_length} ({count})")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Advanced analytics dashboard
                    analytics = generate_advanced_analytics(results)
                    st.markdown("### 📊 Advanced Analytics")

                    # Create a dashboard card layout
                    dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)
                    with dashboard_col1:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown("### 🔒 Security")
                        st.metric("Strength", f"{analytics.get('security_strength', 0):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with dashboard_col2:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown("### 📋 Compliance")
                        st.metric("Coverage", f"{analytics.get('compliance_coverage', 0):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with dashboard_col3:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown("### 🧾 Clarity")
                        st.metric("Score", f"{analytics.get('clarity_score', 0):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with dashboard_col4:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown("### ✅ Completeness")
                        st.metric("Score", f"{analytics.get('completeness_score', 0):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Efficiency and resolution metrics
                    efficiency_col1, efficiency_col2 = st.columns(2)
                    with efficiency_col1:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown("### ⚡ Efficiency")
                        st.metric("Score", f"{analytics.get('efficiency_score', 0):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with efficiency_col2:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Resolution")
                        st.metric("Rate", f"{analytics.get('issue_resolution_rate', 0):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Detailed metrics tabs
                    metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(
                        ["📈 Confidence Trend", "🏆 Model Performance", "🧮 Issue Analysis"])

                    with metrics_tab1:
                        # Confidence trend chart
                        if results.get('iterations'):
                            confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0)
                                                  for iter in results.get('iterations', [])]
                            if confidence_history:
                                # Enhanced visualization
                                max_confidence = max(confidence_history)
                                min_confidence = min(confidence_history)
                                avg_confidence = sum(confidence_history) / len(confidence_history)

                                st.line_chart(confidence_history)
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("📈 Peak Confidence", f"{max_confidence:.1f}%")
                                col2.metric("📉 Lowest Confidence", f"{min_confidence:.1f}%")
                                col3.metric("📊 Average Confidence", f"{avg_confidence:.1f}%")
                                col4.metric("📊 Final Confidence", f"{confidence_history[-1]:.1f}%")

                                # Confidence improvement
                                if len(confidence_history) > 1:
                                    improvement = confidence_history[-1] - confidence_history[0]
                                    if improvement > 0:
                                        st.success(f"🚀 Confidence improved by {improvement:.1f}%")
                                    elif improvement < 0:
                                        st.warning(f"⚠️ Confidence decreased by {abs(improvement):.1f}%")
                                    else:
                                        st.info("➡️ Confidence remained stable")

                    with metrics_tab2:
                        # Model performance analysis
                        if st.session_state.get("adversarial_model_performance"):
                            model_performance = st.session_state.adversarial_model_performance
                            st.markdown("### 🏆 Top Performing Models")

                            # Sort models by score
                            sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get("score", 0),
                                                   reverse=True)

                            # Display top 5 models with enhanced visualization
                            for i, (model_id, perf) in enumerate(sorted_models[:5]):
                                score = perf.get("score", 0)
                                issues = perf.get("issues_found", 0)
                                st.progress(min(score / 100, 1.0),
                                            text=f"#{i + 1} {model_id} - Score: {score}, Issues Found: {issues}")
                        else:
                            st.info("No model performance data available.")

                    with metrics_tab3:
                        # Issue analysis
                        if results.get('iterations'):
                            # Aggregate issue data
                            total_issues = 0
                            severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                            category_counts = {}

                            for iteration in results.get('iterations', []):
                                critiques = iteration.get("critiques", [])
                                for critique in critiques:
                                    critique_json = critique.get("critique_json", {})
                                    issues = critique_json.get("issues", [])
                                    total_issues += len(issues)

                                    for issue in issues:
                                        # Count by severity
                                        severity = issue.get("severity", "low").lower()
                                        if severity in severity_counts:
                                            severity_counts[severity] += 1

                                        # Count by category
                                        category = issue.get("category", "uncategorized")
                                        category_counts[category] = category_counts.get(category, 0) + 1

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### 🎯 Issue Severity Distribution")
                                for severity, count in severity_counts.items():
                                    if count > 0:
                                        emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[severity]
                                        st.write(f"{emoji} {severity.capitalize()}: {count}")
                            with col2:
                                st.markdown("### 📚 Issue Categories")
                                # Show top 5 categories
                                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                                for category, count in sorted_categories[:5]:
                                    st.write(f"🏷️ {category}: {count}")

                            st.metric("🔍 Total Issues Found", total_issues)

                    # Protocol comparison and analysis
                    st.markdown("### 📄 Protocol Analysis")
                    final_sop = results.get('final_sop', '')
                    original_sop = st.session_state.protocol_text

                    if final_sop and original_sop:
                        # Tabs for different views
                        protocol_tab1, protocol_tab2, protocol_tab3, protocol_tab4 = st.tabs(
                            ["🔄 Comparison", "📄 Final Protocol", "🔍 Structure Analysis", "🤖 AI Insights"])

                        with protocol_tab1:
                            st.markdown("### 🔄 Protocol Evolution")
                            # Simple comparison metrics
                            original_complexity = calculate_protocol_complexity(original_sop)
                            final_complexity = calculate_protocol_complexity(final_sop)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("📏 Original Length", f"{original_complexity['word_count']} words")
                            col2.metric("📏 Final Length", f"{final_complexity['word_count']} words")
                            col3.metric("📊 Length Change",
                                        f"{final_complexity['word_count'] - original_complexity['word_count']} words",
                                        f"{((final_complexity['word_count'] / max(1, original_complexity['word_count'])) - 1) * 100:.1f}%")

                            # Show both protocols side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original Protocol**")
                                st.text_area("Original", value=original_sop, height=300,
                                             key="original_protocol_display")
                            with col2:
                                st.markdown("**Hardened Protocol**")
                                st.text_area("Final", value=final_sop, height=300, key="final_protocol_display")

                            # Add detailed comparison
                            st.markdown("### 📊 Detailed Comparison")
                            st.markdown(render_protocol_comparison(original_sop, final_sop, "Original Protocol",
                                                                   "Hardened Protocol"), unsafe_allow_html=True)

                        with protocol_tab2:
                            st.markdown("### 📄 Final Hardened Protocol")
                            st.code(final_sop, language="markdown")
                            # Add download button
                            st.download_button(
                                label="📥 Download Final Protocol",
                                data=final_sop,
                                file_name="hardened_protocol.md",
                                mime="text/markdown"
                            )

                        with protocol_tab3:
                            st.markdown("### 🔍 Protocol Structure Analysis")
                            # Add protocol analysis
                            complexity = calculate_protocol_complexity(final_sop)
                            structure = extract_protocol_structure(final_sop)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📏 Complexity Metrics**")
                                st.metric("Words", complexity["word_count"])
                                st.metric("Sentences", complexity["sentence_count"])
                                st.metric("Paragraphs", complexity["paragraph_count"])
                                st.metric("Complexity Score", complexity["complexity_score"])
                                st.metric("Unique Words", complexity["unique_words"])

                            with col2:
                                st.markdown("**🧩 Structure Analysis**")
                                st.write("Numbered Steps:", "✅" if structure["has_numbered_steps"] else "❌")
                                st.write("Bullet Points:", "✅" if structure["has_bullet_points"] else "❌")
                                st.write("Headers:", "✅" if structure["has_headers"] else "❌")
                                st.write("Preconditions:", "✅" if structure["has_preconditions"] else "❌")
                                st.write("Postconditions:", "✅" if structure["has_postconditions"] else "❌")
                                st.write("Error Handling:", "✅" if structure["has_error_handling"] else "❌")
                                st.metric("SectionsIn Protocol", structure["section_count"])

                        with protocol_tab4:
                            st.markdown("### 🤖 AI Insights Dashboard")
                            st.markdown(render_ai_insights_dashboard(final_sop), unsafe_allow_html=True)


with tab2:
    render_adversarial_testing_tab()

with tab3:
    st.title("🐙 GitHub Integration")
    
    # Authentication check
    if not st.session_state.get("github_token"):
        st.warning("Please authenticate with GitHub in the sidebar first.")
        st.info("Go to the sidebar and enter your GitHub Personal Access Token to get started.")
        st.stop()
    
    # Repository selection
    linked_repos = list_linked_github_repositories()
    if not linked_repos:
        st.warning("Please link at least one GitHub repository in the sidebar first.")
        st.info("Go to the sidebar, find the GitHub Integration section, and link a repository.")
        st.stop()
    
    selected_repo = st.selectbox("Select Repository", linked_repos)
    
    # Branch management
    if selected_repo:
        st.markdown("### 🌿 Branch Management")
        
        # Create new branch
        with st.expander("Create New Branch"):
            new_branch_name = st.text_input("New Branch Name", placeholder="e.g., protocol-v1")
            base_branch = st.text_input("Base Branch", "main")
            if st.button("Create Branch") and new_branch_name:
                token = st.session_state.github_token
                if create_github_branch(token, selected_repo, new_branch_name, base_branch):
                    st.success(f"Created branch '{new_branch_name}' from '{base_branch}'")
        
        # Branch selection
        branch_name = st.text_input("Target Branch", "main")
        
        # Commit and push
        st.markdown("### 💾 Commit and Push")
        file_path = st.text_input("File Path", "protocols/evolved_protocol.md")
        commit_message = st.text_input("Commit Message", "Update evolved protocol")
        
        if st.button("Commit to GitHub") and st.session_state.protocol_text.strip():
            token = st.session_state.github_token
            if commit_to_github(token, selected_repo, file_path, st.session_state.protocol_text, commit_message, branch_name):
                st.success("✅ Committed to GitHub successfully!")
            else:
                st.error("❌ Failed to commit to GitHub")

# Add start/stop functionality for adversarial testingxt, commit_message, branch_name):\n                st.success(\"Successfully committed to GitHub!\")\n                \n                # Store generation info\n                if \"github_generations\" not in st.session_state:\n                    st.session_state.github_generations = []\n                st.session_state.github_generations.append({\n                    \"repo\": selected_repo,\n                    \"file_path\": file_path,\n                    \"branch\": branch_name,\n                    \"timestamp\": datetime.now().isoformat(),\n                    \"commit_message\": commit_message\n                })\n        \n        # View commit history\n        st.markdown(\"### 📜 Commit History\")\n        if st.button(\"Fetch Commit History\"):\n            token = st.session_state.github_token\n            commits = get_github_commit_history(token, selected_repo, file_path, branch_name)\n            if commits:\n                for commit in commits:\n                    col1, col2, col3 = st.columns([2, 1, 1])\n                    with col1:\n                        st.caption(f\"**{commit['message'][:50]}{'...' if len(commit['message']) > 50 else ''}**\")\n                    with col2:\n                        st.caption(f\"{commit['author']}\")\n                    with col3:\n                        st.caption(f\"{commit['date'][:10]}\")\n            else:\n                st.info(\"No commits found for this file.\")\n        \n        # Protocol generations\n        st.markdown(\"### 📚 Protocol Generations\")\n        generations = get_protocol_generations_from_github(selected_repo)\n        if generations:\n            for gen in generations:\n                col1, col2, col3 = st.columns([2, 1, 1])\n                with col1:\n                    st.caption(f\"**{gen['generation_name']}**\")\n                with col2:\n                    st.caption(f\"{gen['timestamp'][:10]}\")\n                with col3:\n                    if st.button(\"View\", key=f\"view_gen_{gen['file_path']}\"):\n                        st.code(f\"# {gen['generation_name']}\\n\\n{st.session_state.protocol_text}\", language=\"markdown\")\n        else:\n            st.info(\"No protocol generations found in this repository.\")\n\n# Add start/stop functionality for adversarial testing
if start_button:
    # Validate inputs before starting
    errors = []

    if not st.session_state.protocol_text.strip():
        errors.append("📄 Please enter a protocol before starting adversarial testing.")

    if not st.session_state.openrouter_key:
        errors.append("🔑 Please enter your OpenRouter API key.")

    if not st.session_state.red_team_models:
        errors.append("🔴 Please select at least one red team model.")

    if not st.session_state.blue_team_models:
        errors.append("🔵 Please select at least one blue team model.")

    if st.session_state.adversarial_min_iter > st.session_state.adversarial_max_iter:
        errors.append("🔄 Min iterations cannot be greater than max iterations.")

    if st.session_state.adversarial_confidence < 50 or st.session_state.adversarial_confidence > 100:
        errors.append("🎯 Confidence threshold should be between 50% and 100%.")

    # Enhanced validation for model selection
    if len(st.session_state.red_team_models) > 10:
        errors.append("🔴 Please select 10 or fewer red team models to avoid excessive costs.")

    if len(st.session_state.blue_team_models) > 10:
        errors.append("🔵 Please select 10 or fewer blue team models to avoid excessive costs.")

    # Show all errors at once
    if errors:
        for error in errors:
            st.error(error)
        st.info("💡 Tip: Check the Quick Start Wizard above for guided setup.")
    else:
        # Thread safety check to prevent multiple concurrent adversarial testing threads
        if st.session_state.adversarial_running:
            st.warning("Adversarial testing is already running. Please wait for it to complete or stop it first.")
        else:
            # Enhanced confirmation dialog with detailed cost estimation
            protocol_length = len(st.session_state.protocol_text.split())
            estimate = estimate_testing_time_and_cost(
                st.session_state.red_team_models,
                st.session_state.blue_team_models,
                st.session_state.adversarial_max_iter,
                protocol_length
            )

            estimated_cost = estimate["estimated_cost_usd"]
            estimated_time = estimate["estimated_time_minutes"]

            if estimated_cost > 0.1:  # If estimated cost is over $0.10
                st.warning(
                    f"💰 Estimated cost: ${estimated_cost:.4f} | ⏰ Estimated time: {estimated_time} minutes | 🔄 Operations: {estimate['total_operations']:,}")
            st.info("💡 Tip: Consider reducing the number of models or iterations to control costs.")
            if not st.checkbox("✅ I understand the cost and time estimate and want to proceed", key="cost_confirmation"):
                st.info("ℹ️ Please confirm you understand the cost and time estimate to proceed.")
                st.stop()

            # Start adversarial testing with enhanced initialization
            st.session_state.adversarial_running = True
            st.session_state.adversarial_status_message = "🚀 Initializing adversarial testing..."

            # Show initialization message
            with st.spinner("Starting adversarial testing process..."):
                # Initialize progress tracking
                st.session_state.adversarial_confidence_history = []
                st.session_state.adversarial_total_tokens_prompt = 0
                st.session_state.adversarial_total_tokens_completion = 0
                st.session_state.adversarial_cost_estimate_usd = 0.0

                # Start the testing process in a separate thread
                threading.Thread(target=run_adversarial_testing, daemon=True).start()
            st.rerun()

if stop_button:
    if st.session_state.adversarial_running:
        st.session_state.adversarial_stop_flag = True
        st.warning("⏹️ Stop signal sent. Adversarial testing will stop after the current iteration.")
        st.session_state.adversarial_status_message = "⏹️ Stopping adversarial testing..."
        # No rerun here, let the loop handle the UI update
    else:
        st.info("ℹ️ Adversarial testing is not currently running.")
