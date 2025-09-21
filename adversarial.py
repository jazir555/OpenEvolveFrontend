# ------------------------------------------------------------------
# 5. Adversarial Testing Functions (robust)
# ------------------------------------------------------------------

MODEL_META_BY_ID: Dict[str, Dict[str, Any]] = {}
MODEL_META_LOCK = threading.Lock()


@st.cache_data(ttl=600)
def get_openrouter_models(api_key: str) -> List[Dict]:
    """Fetch available models from OpenRouter (cached)."""
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return models
    except Exception as e:
        st.warning(f"Could not fetch OpenRouter models: {e}")
        return []


def _compose_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _request_openrouter_chat(
        api_key: str,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        max_tokens: int,
        force_json: bool = False,
        seed: Optional[int] = None,
        req_timeout: int = 60,
        max_retries: int = 5,
) -> Tuple[str, int, int, float]:
    """
    Robust OpenRouter chat call with exponential backoff, jitter, and cost/token estimation.
    Returns: (content, prompt_tokens, completion_tokens, cost)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/google/gemini-pro-builder",  # Recommended by OpenRouter
        "X-Title": "OpenEvolve Protocol Improver",  # Recommended by OpenRouter
    }
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": _clamp(temperature, 0.0, 2.0),
        "max_tokens": max_tokens,
        "top_p": _clamp(top_p, 0.0, 1.0),
        "frequency_penalty": _clamp(frequency_penalty, -2.0, 2.0),
        "presence_penalty": _clamp(presence_penalty, -2.0, 2.0),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code == 400:
                # HTTP 400 Bad Request - client error, don't retry
                last_err = Exception(f"HTTP 400 Bad Request: {r.text[:200]}...")
                break  # Break out of retry loop for client errors
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"Transient error {r.status_code}: Retrying...")
                continue
            r.raise_for_status()
            data = r.json()

            # Safely access the first choice to prevent IndexError if "choices" is an empty list.
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
            else:
                content = ""

            usage = data.get("usage", {})
            p_tok = safe_int(usage.get("prompt_tokens"), _approx_tokens(json.dumps(messages)))
            c_tok = safe_int(usage.get("completion_tokens"), _approx_tokens(content or ""))
            cost = _cost_estimate(p_tok, c_tok, None, None)  # Simplified cost calculation
            return content or "", p_tok, c_tok, cost
        except Exception as e:
            last_err = e
            sleep_s = (2 ** attempt) + _rand_jitter_ms()
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed for {model_id} after {max_retries} attempts: {last_err}")


def _request_anthropic_chat(
        api_key: str, base_url: str, model: str, messages: List, extra_headers: Dict,
        temperature: float, top_p: float, max_tokens: int, seed: Optional[int],
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
        req_timeout: int = 60, max_retries: int = 5
) -> str:
    url = base_url.rstrip('/') + "/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        **extra_headers
    }

    # Separate system prompt from messages
    system_prompt = ""
    user_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            system_prompt = msg['content']
        else:
            user_messages.append(msg)

    payload = {
        "model": model,
        "messages": user_messages,
        "max_tokens": max_tokens,
        "temperature": _clamp(temperature, 0.0, 1.0),
        "top_p": _clamp(top_p, 0.0, 1.0),
    }
    if system_prompt:
        payload['system'] = system_prompt

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"HTTP {r.status_code}: {r.text}")
                continue
            r.raise_for_status()
            data = r.json()
            if data.get('content') and isinstance(data['content'], list):
                return data['content'][0].get('text', '')
            else:
                last_err = Exception("No content in response")
        except Exception as e:
            last_err = e
            sleep_s = (2 ** attempt) + _rand_jitter_ms()
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {max_retries} attempts for model {model}: {last_err}")


def _request_google_gemini_chat(
        api_key: str, base_url: str, model: str, messages: List, extra_headers: Dict,
        temperature: float, top_p: float, max_tokens: int, seed: Optional[int],
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
        req_timeout: int = 60, max_retries: int = 5
) -> str:
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json", **extra_headers}

    # Gemini uses a different message format
    contents = []
    for msg in messages:
        contents.append({"role": msg['role'], "parts": [{"text": msg['content']}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": _clamp(temperature, 0.0, 1.0),
            "topP": _clamp(top_p, 0.0, 1.0),
            "maxOutputTokens": max_tokens,
        }
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"HTTP {r.status_code}: {r.text}")
                continue
            r.raise_for_status()
            data = r.json()
            if data.get('candidates') and isinstance(data['candidates'], list):
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                last_err = Exception("No content in response")
        except Exception as e:
            last_err = e
            sleep_s = (2 ** attempt) + _rand_jitter_ms()
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {max_retries} attempts for model {model}: {last_err}")


def _request_cohere_chat(
        api_key: str, base_url: str, model: str, messages: List, extra_headers: Dict,
        temperature: float, top_p: float, max_tokens: int, seed: Optional[int],
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
        req_timeout: int = 60, max_retries: int = 5
) -> str:
    url = base_url.rstrip('/') + "/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **extra_headers
    }

    # Separate system prompt and history from the last user message
    system_prompt = ""
    chat_history = []
    for msg in messages[:-1]:
        if msg['role'] == 'system':
            system_prompt = msg['content']
        else:
            chat_history.append({"role": msg['role'].upper(), "message": msg['content']})

    user_message = messages[-1]['content']

    payload = {
        "model": model,
        "message": user_message,
        "chat_history": chat_history,
        "max_tokens": max_tokens,
        "temperature": _clamp(temperature, 0.0, 5.0),
        "p": _clamp(top_p, 0.0, 1.0),
    }
    if system_prompt:
        payload['preamble'] = system_prompt

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"HTTP {r.status_code}: {r.text}")
                continue
            r.raise_for_status()
            data = r.json()
            if data.get('text'):
                return data['text']
            else:
                last_err = Exception("No text in response")
        except Exception as e:
            last_err = e
            sleep_s = (2 ** attempt) + _rand_jitter_ms()
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {max_retries} attempts for model {model}: {last_err}")


def analyze_with_model(
        api_key: str,
        model_id: str,
        sop: str,
        config: Dict,
        system_prompt: str,
        user_suffix: str = "",
        force_json: bool = False,
        seed: Optional[int] = None,
        compliance_requirements: str = "",
) -> Dict[str, Any]:
    """
    Analyzes an SOP with a specific model, handling context limits and returning structured results.
    """
    try:
        if compliance_requirements:
            system_prompt = system_prompt.format(compliance_requirements=compliance_requirements)
        max_tokens = safe_int(config.get("max_tokens"), 8000)
        user_prompt = f"Here is the Standard Operating Procedure (SOP):\n\n---\n\n{sop}\n\n---\n\n{user_suffix}"
        full_prompt_text = system_prompt + user_prompt

        # Simplified context length estimation
        context_len = 8192
        prompt_toks_est = _approx_tokens(full_prompt_text)

        if prompt_toks_est + max_tokens >= context_len:
            err_msg = (f"ERROR[{model_id}]: Estimated prompt tokens ({prompt_toks_est}) + max_tokens ({max_tokens}) "
                       f"exceeds context window ({context_len}). Skipping.")
            return {"ok": False, "text": err_msg, "json": None, "ptoks": 0, "ctoks": 0, "cost": 0.0,
                    "model_id": model_id}

        content, p_tok, c_tok, cost = _request_openrouter_chat(
            api_key=api_key, model_id=model_id,
            messages=_compose_messages(system_prompt, user_prompt),
            temperature=safe_float(config.get("temperature"), 0.7),
            top_p=safe_float(config.get("top_p"), 1.0),
            frequency_penalty=safe_float(config.get("frequency_penalty"), 0.0),
            presence_penalty=safe_float(config.get("presence_penalty"), 0.0),
            max_tokens=max_tokens, force_json=force_json, seed=seed,
        )
        json_content = _extract_json_block(content)
        return {"ok": True, "text": content, "json": json_content, "ptoks": p_tok, "ctoks": c_tok, "cost": cost,
                "model_id": model_id}
    except Exception as e:
        return {"ok": False, "text": f"ERROR[{model_id}]: {e}", "json": None, "ptoks": 0, "ctoks": 0, "cost": 0.0,
                "model_id": model_id}


def determine_review_type(content: str) -> str:
    """Determine the appropriate review type based on content analysis.

    Args:
        content (str): The content to analyze

    Returns:
        str: Review type ('general', 'code', 'plan')
    """
    if not content:
        return "general"

    # Convert to lowercase for analysis
    lower_content = content.lower()

    # Check for code indicators
    code_indicators = [
        'function ', 'def ', 'class ', 'import ', 'require(', 'var ', 'let ', 'const ',
        'public ', 'private ', 'protected ', 'static ', 'void ', 'int ', 'string ',
        '<html', '<?php', '<script', 'console.', 'print(', 'printf(', 'scanf(',
        'if(', 'for(', 'while(', 'switch(', 'try{', 'catch(', 'finally{'
    ]

    # Check for plan indicators
    plan_indicators = [
        'objective', 'goal', 'milestone', 'deliverable', 'resource', 'budget',
        'timeline', 'schedule', 'risk', 'dependency', 'assumption',
        'stakeholder', 'communication', 'review', 'approval'
    ]

    # Count matches
    code_matches = sum(1 for indicator in code_indicators if indicator in lower_content)
    plan_matches = sum(1 for indicator in plan_indicators if indicator in lower_content)

    # Determine review type
    if code_matches > plan_matches and code_matches > 2:
        return "code"
    elif plan_matches > code_matches and plan_matches > 2:
        return "plan"
    else:
        return "general"


def get_appropriate_prompts(review_type: str) -> Tuple[str, str]:
    """Get the appropriate prompts based on review type.

    Args:
        review_type (str): Type of review ('general', 'code', 'plan')

    Returns:
        Tuple[str, str]: Red team and blue team prompts
    """
    if review_type == "code":
        return CODE_REVIEW_RED_TEAM_PROMPT, CODE_REVIEW_BLUE_TEAM_PROMPT
    elif review_type == "plan":
        return PLAN_REVIEW_RED_TEAM_PROMPT, PLAN_REVIEW_BLUE_TEAM_PROMPT
    else:
        return RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT


def _severity_rank(sev: str) -> int:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return order.get(str(sev).lower(), 0)


def _merge_consensus_sop(base_sop: str, blue_patches: List[dict], critiques: List[dict]) -> Tuple[str, dict]:
    """
    Selects the best patch from the blue team based on coverage, resolution, and quality.
    """
    valid_patches = [p for p in blue_patches if p and (p.get("patch_json") or {}).get("sop", "").strip()]
    if not valid_patches:
        return base_sop, {"reason": "no_valid_patches_received", "score": -1, "resolution_by_severity": {},
                          "resolution_by_category": {}}

    # Create a lookup for issue severity and category
    issue_details = {}
    for critique in critiques:
        if critique and critique.get("critique_json"):
            for issue in _safe_list(critique["critique_json"], "issues"):
                issue_details[issue.get("title")] = {
                    "severity": issue.get("severity", "low"),
                    "category": issue.get("category", "uncategorized")
                }

    scored = []
    for patch in valid_patches:
        patch_json = patch.get("patch_json", {})
        sop_text = patch_json.get("sop", "").strip()

        mm = _safe_list(patch_json, "mitigation_matrix")
        residual = _safe_list(patch_json, "residual_risks")

        resolved = sum(1 for r in mm if str(r.get("status", "")).lower() == "resolved")
        mitigated = sum(1 for r in mm if str(r.get("status", "")).lower() == "mitigated")

        # Score based on resolved issues, then mitigated, penalize for residuals, and use length as tie-breaker
        coverage_score = (resolved * 2) + mitigated
        final_score = coverage_score - (len(residual) * 2)

        # Track resolution by severity and category
        resolution_by_severity = {}
        resolution_by_category = {}
        for r in mm:
            issue_title = r.get("issue")
            if issue_title in issue_details:
                details = issue_details[issue_title]
                severity = details["severity"]
                category = details["category"]
                status = str(r.get("status", "")).lower()

                if status in ["resolved", "mitigated"]:
                    resolution_by_severity[severity] = resolution_by_severity.get(severity, 0) + 1
                    resolution_by_category[category] = resolution_by_category.get(category, 0) + 1

        scored.append((final_score, resolved, len(sop_text), sop_text, patch.get("model"), resolution_by_severity,
                       resolution_by_category))

    if not scored:
        return base_sop, {"reason": "all_patches_were_empty_or_invalid", "score": -1, "resolution_by_severity": {},
                          "resolution_by_category": {}}

    # Sort by score, then resolved count, then SOP length
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    best_score, best_resolved, _, best_sop, best_model, best_res_sev, best_res_cat = scored[0]
    diagnostics = {"reason": "best_patch_selected", "score": best_score, "resolved": best_resolved, "model": best_model,
                   "resolution_by_severity": best_res_sev, "resolution_by_category": best_res_cat}
    return best_sop, diagnostics


def _aggregate_red_risk(critiques: List[dict]) -> Dict[str, Any]:
    """Computes an aggregate risk score from all red-team critiques."""
    sev_weight = {"low": 1, "medium": 3, "high": 6, "critical": 12}
    total_weight, issue_count = 0, 0
    categories = {}
    severities = {}

    valid_critiques = [c.get("critique_json") for c in critiques if c and c.get("critique_json")]

    for critique in valid_critiques:
        for issue in _safe_list(critique, "issues"):
            sev = str(issue.get("severity", "low")).lower()
            weight = sev_weight.get(sev, 1)
            total_weight += weight
            issue_count += 1
            cat = str(issue.get("category", "uncategorized")).lower()
            categories[cat] = categories.get(cat, 0) + weight
            severities[sev] = severities.get(sev, 0) + 1

    avg_weight = (total_weight / max(1, issue_count)) if issue_count > 0 else 0
    return {"total_weight": total_weight, "avg_issue_weight": avg_weight, "categories": categories,
            "severities": severities, "count": issue_count}


def _update_model_performance(critiques: List[dict]):
    """Updates the performance scores of models based on the critiques they generated."""
    with st.session_state.thread_lock:
        if "adversarial_model_performance" not in st.session_state:
            st.session_state.adversarial_model_performance = {}

        sev_weight = {"low": 1, "medium": 3, "high": 6, "critical": 12}
        for critique in critiques:
            model_id = critique.get("model")
            if not model_id:
                continue

            if model_id not in st.session_state.adversarial_model_performance:
                st.session_state.adversarial_model_performance[model_id] = {"score": 0, "issues_found": 0}

            critique_json = critique.get("critique_json")
            if critique_json and isinstance(critique_json.get("issues"), list):
                for issue in critique_json["issues"]:
                    sev = str(issue.get("severity", "low")).lower()
                    st.session_state.adversarial_model_performance[model_id]["score"] += sev_weight.get(sev, 1)
                    st.session_state.adversarial_model_performance[model_id]["issues_found"] += 1


def _collect_model_configs(model_ids: List[str], max_tokens: int) -> Dict[str, Dict[str, Any]]:
    return {
        model_id: {
            "temperature": st.session_state.get(f"temp_{model_id}", 0.7),
            "top_p": st.session_state.get(f"topp_{model_id}", 1.0),
            "frequency_penalty": st.session_state.get(f"freqpen_{model_id}", 0.0),
            "presence_penalty": st.session_state.get(f"prespen_{model_id}", 0.0),
            "max_tokens": max_tokens,
        } for model_id in model_ids
    }


def _update_adv_log_and_status(msg: str):
    """Thread-safe way to update logs and status message."""
    with st.session_state.thread_lock:
        st.session_state.adversarial_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        st.session_state.adversarial_status_message = msg


def _update_adv_counters(ptoks: int, ctoks: int, cost: float):
    """Thread-safe way to update token and cost counters."""
    with st.session_state.thread_lock:
        st.session_state.adversarial_total_tokens_prompt += ptoks
        st.session_state.adversarial_total_tokens_completion += ctoks
        st.session_state.adversarial_cost_estimate_usd += cost


def check_approval_rate(
        api_key: str, red_team_models: List[str], sop_markdown: str, model_configs: Dict,
        seed: Optional[int], max_workers: int, approval_prompt: str = APPROVAL_PROMPT
) -> Dict[str, Any]:
    """Asks all red-team models for a final verdict on the SOP."""
    votes, scores, approved = [], [], 0
    total_ptoks, total_ctoks, total_cost = 0, 0, 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_model = {
            ex.submit(
                analyze_with_model, api_key, model_id, sop_markdown,
                model_configs.get(model_id, {}), approval_prompt, force_json=True, seed=seed
            ): model_id for model_id in red_team_models
        }
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            res = future.result()
            total_ptoks += res["ptoks"];
            total_ctoks += res["ctoks"];
            total_cost += res["cost"]
            if res.get("ok") and res.get("json"):
                j = res["json"]
                verdict = str(j.get("verdict", "REJECTED")).upper()
                score = _clamp(safe_int(j.get("score"), 0), 0, 100)
                if verdict == "APPROVED":
                    approved += 1
                scores.append(score)
                votes.append(
                    {"model": model_id, "verdict": verdict, "score": score, "reasons": _safe_list(j, "reasons")})
            else:
                votes.append({"model": model_id, "verdict": "ERROR", "score": 0, "reasons": [res.get("text")]})

    rate = (approved / max(1, len(red_team_models))) * 100.0
    avg_score = (sum(scores) / max(1, len(scores))) if scores else 0

    # Calculate agreement
    if not votes:
        agreement = 0.0
    else:
        verdicts = [v["verdict"] for v in votes]
        approved_count = verdicts.count("APPROVED")
        rejected_count = verdicts.count("REJECTED")
        agreement = max(approved_count, rejected_count) / len(verdicts) * 100.0

    return {"approval_rate": rate, "avg_score": avg_score, "votes": votes, "prompt_tokens": total_ptoks,
            "completion_tokens": total_ctoks, "cost": total_cost, "agreement": agreement}


def generate_docx_report(results: dict) -> bytes:
    """Generates a DOCX report from the adversarial testing results."""
    document = docx.Document()
    document.add_heading('Adversarial Testing Report', 0)

    document.add_heading('Summary', level=1)
    document.add_paragraph(
        f"Final Approval Rate: {results.get('final_approval_rate', 0.0):.1f}%\n"
        f"Total Iterations: {len(results.get('iterations', []))}\n"
        f"Total Cost (USD): ${results.get('cost_estimate_usd', 0.0):,.4f}\n"
        f"Total Prompt Tokens: {results.get('tokens', {}).get('prompt', 0):,}\n"
        f"Total Completion Tokens: {results.get('tokens', {}).get('completion', 0):,}"
    )

    document.add_heading('Final Hardened SOP', level=1)
    document.add_paragraph(results.get("final_sop", ""))

    document.add_heading('Issues Found', level=1)
    for i, iteration in enumerate(results.get("iterations", [])):
        document.add_heading(f"Iteration {i + 1}", level=2)
        for critique in iteration.get("critiques", []):
            if critique.get("critique_json"):
                for issue in _safe_list(critique["critique_json"], "issues"):
                    document.add_paragraph(f"- {issue.get('title')} ({issue.get('severity')})", style='List Bullet')

    document.add_heading('Final Votes', level=1)
    if results.get("iterations"):
        for vote in results["iterations"][-1].get("approval_check", {}).get("votes", []):
            document.add_paragraph(f"- {vote.get('model')}: {vote.get('verdict')} ({vote.get('score')})",
                                   style='List Bullet')

    document.add_heading('Audit Trail', level=1)
    for log_entry in results.get("log", []):
        document.add_paragraph(log_entry)

    from io import BytesIO
    bio = BytesIO()
    document.save(bio)
    return bio.getvalue()


def generate_pdf_report(results: dict) -> bytes:
    """Generates a PDF report from the adversarial testing results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Adversarial Testing Report", ln=True, align='C')

    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Final Approval Rate: {results.get('final_approval_rate', 0.0):.1f}%\n"
                          f"Total Iterations: {len(results.get('iterations', []))}\n"
                          f"Total Cost (USD): ${results.get('cost_estimate_usd', 0.0):,.4f}\n"
                          f"Total Prompt Tokens: {results.get('tokens', {}).get('prompt', 0):,}\n"
                          f"Total Completion Tokens: {results.get('tokens', {}).get('completion', 0):,}")

    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Final Hardened SOP", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, results.get("final_sop", ""))

    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Issues Found", ln=True)
    pdf.set_font("Arial", size=12)
    for i, iteration in enumerate(results.get("iterations", [])):
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(200, 10, txt=f"Iteration {i + 1}", ln=True)
        pdf.set_font("Arial", size=10)
        for critique in iteration.get("critiques", []):
            if critique.get("critique_json"):
                for issue in _safe_list(critique["critique_json"], "issues"):
                    pdf.multi_cell(0, 10, f"- {issue.get('title')} ({issue.get('severity')})")

    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Final Votes", ln=True)
    pdf.set_font("Arial", size=10)
    if results.get("iterations"):
        for vote in results["iterations"][-1].get("approval_check", {}).get("votes", []):
            pdf.multi_cell(0, 10, f"- {vote.get('model')}: {vote.get('verdict')} ({vote.get('score')})")

    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Audit Trail", ln=True)
    pdf.set_font("Arial", size=8)
    for log_entry in results.get("log", []):
        pdf.multi_cell(0, 5, log_entry)

    return pdf.output(dest='S').encode('latin-1')


def generate_html_report(results: dict) -> str:
    """Generates an HTML report from the adversarial testing results."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Adversarial Testing Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #4a6fa5; }}
            .summary {{ background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }}
            .section {{ margin: 20px 0; background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
            .log {{ font-family: monospace; font-size: 0.9em; background-color: #f9f9f9; padding: 10px; border-radius: 4px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4a6fa5; color: white; }}
            .metric {{ text-align: center; padding: 10px; background-color: #e9ecef; border-radius: 4px; margin: 5px; }}
            .improvement {{ color: #4caf50; font-weight: bold; }}
            .regression {{ color: #f44336; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Adversarial Testing Report</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Final Approval Rate:</strong> {results.get('final_approval_rate', 0.0):.1f}%</p>
            <p><strong>Total Iterations:</strong> {len(results.get('iterations', []))}</p>
            <p><strong>Total Cost (USD):</strong> ${results.get('cost_estimate_usd', 0.0):,.4f}</p>
            <p><strong>Total Prompt Tokens:</strong> {results.get('tokens', {}).get('prompt', 0):,}</p>
            <p><strong>Total Completion Tokens:</strong> {results.get('tokens', {}).get('completion', 0):,}</p>
        </div>

        <div class="section">
            <h2>Final Hardened SOP</h2>
            <pre style="white-space: pre-wrap; background-color: #f9f9f9; padding: 15px; border-radius: 4px;">{results.get("final_sop", "")}</pre>
        </div>
    """

    if results.get("iterations"):
        html += """
        <div class="section">
            <h2>Issues Found</h2>
        """
        for i, iteration in enumerate(results.get("iterations", [])):
            html += f"<h3>Iteration {i + 1}</h3><ul>"
            for critique in iteration.get("critiques", []):
                if critique.get("critique_json"):
                    for issue in _safe_list(critique["critique_json"], "issues"):
                        severity = issue.get('severity', 'low')
                        severity_color = {
                            'low': '#4caf50',
                            'medium': '#ff9800',
                            'high': '#f44336',
                            'critical': '#9c27b0'
                        }.get(severity, '#000000')
                        html += f"<li><span style='color: {severity_color}; font-weight: bold;'>{severity.upper()}</span>: {issue.get('title')}</li>"
            html += "</ul>"
        html += "</div>"

        html += """
        <div class="section">
            <h2>Final Votes</h2>
            <table>
                <tr><th>Model</th><th>Verdict</th><th>Score</th></tr>
        """
        for vote in results["iterations"][-1].get("approval_check", {}).get("votes", []):
            verdict = vote.get('verdict', '')
            verdict_color = '#4caf50' if verdict.upper() == 'APPROVED' else '#f44336'
            html += f"<tr><td>{vote.get('model')}</td><td style='color: {verdict_color}; font-weight: bold;'>{verdict}</td><td>{vote.get('score')}</td></tr>"
        html += "</table></div>"

        # Add performance chart data
        html += """
        <div class="section">
            <h2>Performance Metrics</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
        """

        # Approval rate chart
        approval_rates = [iter["approval_check"].get("approval_rate", 0) for iter in results.get("iterations", [])]
        if approval_rates:
            avg_approval = sum(approval_rates) / len(approval_rates)
            html += f"""
            <div class="metric">
                <div>Avg Approval Rate</div>
                <div style="font-size: 24px;">{avg_approval:.1f}%</div>
            </div>
            """

        # Issue count chart
        issue_counts = [iter["agg_risk"].get("count", 0) for iter in results.get("iterations", [])]
        if issue_counts:
            total_issues = sum(issue_counts)
            html += f"""
            <div class="metric">
                <div>Total Issues Found</div>
                <div style="font-size: 24px;">{total_issues}</div>
            </div>
            """

        html += "</div></div>"

    html += """
        <div class="section">
            <h2>Audit Trail</h2>
            <div class="log">
    """
    for log_entry in results.get("log", []):
        html += f"<div>{log_entry}</div>"
    html += """
            </div>
        </div>
    </body>
    </html>
    """

    return html


# ------------------------------------------------------------------
# Performance Optimization Functions
# ------------------------------------------------------------------

def optimize_model_selection(red_team_models: List[str], blue_team_models: List[str],
                             protocol_complexity: int, budget_limit: float = 0.0) -> Dict[str, List[str]]:
    """Optimize model selection based on protocol complexity and budget.

    Args:
        red_team_models (List[str]): Available red team models
        blue_team_models (List[str]): Available blue team models
        protocol_complexity (int): Complexity score of the protocol (0-100)
        budget_limit (float): Maximum budget in USD (0 = no limit)

    Returns:
        Dict[str, List[str]]: Optimized model selections
    """
    # Enhanced optimization logic
    optimized = {
        "red_team": [],
        "blue_team": []
    }

    # For complex protocols, use more capable models
    if protocol_complexity > 70:
        # Use premium models for complex protocols
        optimized["red_team"] = [m for m in red_team_models if
                                 "gpt-4" in m or "claude-3-opus" in m or "gemini-1.5-pro" in m][:3]
        optimized["blue_team"] = [m for m in blue_team_models if
                                  "gpt-4" in m or "claude-3-sonnet" in m or "gemini-1.5-pro" in m][:3]
    elif protocol_complexity > 40:
        # Use balanced models for medium complexity
        optimized["red_team"] = [m for m in red_team_models if
                                 "gpt-4" in m or "claude-3-haiku" in m or "gemini-1.5-flash" in m][:3]
        optimized["blue_team"] = [m for m in blue_team_models if
                                  "gpt-4" in m or "claude-3-sonnet" in m or "gemini-1.5-flash" in m][:3]
    else:
        # Use cost-effective models for simple protocols
        optimized["red_team"] = [m for m in red_team_models if
                                 "gpt-4o-mini" in m or "claude-3-haiku" in m or "llama-3-8b" in m][:3]
        optimized["blue_team"] = [m for m in blue_team_models if
                                  "gpt-4o" in m or "claude-3-sonnet" in m or "llama-3-70b" in m][:3]

    # Budget-based optimization
    if budget_limit > 0:
        # Filter out expensive models if budget is constrained
        expensive_models = ["gpt-4", "claude-3-opus", "gemini-1.5-pro"]
        if budget_limit < 0.1:  # Very low budget
            optimized["red_team"] = [m for m in optimized["red_team"] if not any(exp in m for exp in expensive_models)]
            optimized["blue_team"] = [m for m in optimized["blue_team"] if
                                      not any(exp in m for exp in expensive_models)]
        elif budget_limit < 0.5:  # Moderate budget
            # Reduce count of expensive models
            expensive_red = [m for m in optimized["red_team"] if any(exp in m for exp in expensive_models)]
            if len(expensive_red) > 1:
                optimized["red_team"] = [m for m in optimized["red_team"] if m not in expensive_red[1:]]

            expensive_blue = [m for m in optimized["blue_team"] if any(exp in m for exp in expensive_models)]
            if len(expensive_blue) > 1:
                optimized["blue_team"] = [m for m in optimized["blue_team"] if m not in expensive_blue[1:]]

    # If no models matched criteria, use defaults
    if not optimized["red_team"]:
        optimized["red_team"] = red_team_models[:min(3, len(red_team_models))]
    if not optimized["blue_team"]:
        optimized["blue_team"] = blue_team_models[:min(3, len(blue_team_models))]

    # Ensure diversity in model selection
    if len(optimized["red_team"]) < 3 and len(red_team_models) >= 3:
        # Add models from different providers
        providers = set()
        selected_models = []
        for model in red_team_models:
            provider = model.split("/")[0] if "/" in model else model.split("-")[0]
            if provider not in providers or len(selected_models) < 3:
                selected_models.append(model)
                providers.add(provider)
        optimized["red_team"] = selected_models[:3]

    if len(optimized["blue_team"]) < 3 and len(blue_team_models) >= 3:
        # Add models from different providers
        providers = set()
        selected_models = []
        for model in blue_team_models:
            provider = model.split("/")[0] if "/" in model else model.split("-")[0]
            if provider not in providers or len(selected_models) < 3:
                selected_models.append(model)
                providers.add(provider)
        optimized["blue_team"] = selected_models[:3]

    return optimized


def estimate_testing_time_and_cost(red_team_models: List[str], blue_team_models: List[str],
                                   iterations: int, protocol_length: int) -> Dict[str, Any]:
    """Estimate testing time and cost based on configuration.

    Args:
        red_team_models (List[str]): Selected red team models
        blue_team_models (List[str]): Selected blue team models
        iterations (int): Number of iterations
        protocol_length (int): Length of protocol in words

    Returns:
        Dict[str, Any]: Time and cost estimates
    """
    # Simplified estimation logic
    # Base estimates per model per iteration
    avg_response_time = 5  # seconds
    avg_cost_per_1000_tokens = 0.002  # USD

    # Calculate total operations
    total_red_operations = len(red_team_models) * iterations
    total_blue_operations = len(blue_team_models) * iterations

    # Estimate time (parallel processing assumed)
    max_parallel_workers = min(6, len(red_team_models) + len(blue_team_models))
    estimated_time_seconds = ((total_red_operations + total_blue_operations) / max_parallel_workers) * avg_response_time

    # Estimate cost (simplified token estimation)
    avg_tokens_per_operation = protocol_length * 3  # Rough estimate
    total_tokens = (total_red_operations + total_blue_operations) * avg_tokens_per_operation
    estimated_cost = (total_tokens / 1000) * avg_cost_per_1000_tokens

    return {
        "estimated_time_minutes": round(estimated_time_seconds / 60, 1),
        "estimated_cost_usd": round(estimated_cost, 4),
        "total_operations": total_red_operations + total_blue_operations,
        "total_tokens_estimated": total_tokens
    }


def suggest_performance_improvements(current_config: Dict) -> List[str]:
    """Suggest performance improvements for the current configuration.

    Args:
        current_config (Dict): Current adversarial testing configuration

    Returns:
        List[str]: List of suggested improvements
    """
    suggestions = []

    red_models = current_config.get("red_team_models", [])
    blue_models = current_config.get("blue_team_models", [])
    iterations = current_config.get("adversarial_max_iter", 10)
    protocol_text = current_config.get("protocol_text", "")

    # Check for common performance issues
    if len(red_models) > 5:
        suggestions.append("🔴 Reduce red team models to 3-5 for better performance and cost control")

    if len(blue_models) > 5:
        suggestions.append("🔵 Reduce blue team models to 3-5 for better performance and cost control")

    if iterations > 20:
        suggestions.append("🔄 Consider reducing max iterations to 15-20 for faster results")

    if len(protocol_text.split()) > 5000:
        suggestions.append("📄 Your protocol is quite long (>5000 words). Consider breaking it into smaller sections")

    # Check for model diversity
    all_models = red_models + blue_models
    if len(set(all_models)) < len(all_models) * 0.7:
        suggestions.append("🔀 Increase model diversity by selecting models from different providers")

    # Check for expensive model combinations
    expensive_models = [m for m in all_models if "gpt-4" in m or "claude-3-opus" in m]
    if len(expensive_models) > 3:
        suggestions.append("💰 You're using many expensive models. Consider mixing in some cost-effective models")

    # If no suggestions, provide positive feedback
    if not suggestions:
        suggestions.append("✅ Your configuration looks well-balanced for optimal performance!")

    return suggestions


# ------------------------------------------------------------------
# Advanced Testing Strategies
# ------------------------------------------------------------------

def adaptive_testing_strategy(results_history: List[Dict], current_config: Dict) -> Dict[str, Any]:
    """Adapt testing strategy based on historical results.

    Args:
        results_history (List[Dict]): History of previous testing results
        current_config (Dict): Current testing configuration

    Returns:
        Dict[str, Any]: Adapted strategy recommendations
    """
    strategy = {
        "recommended_models": {"red_team": [], "blue_team": []},
        "iteration_adjustments": {},
        "focus_areas": [],
        "confidence_threshold": current_config.get("adversarial_confidence", 85)
    }

    if not results_history:
        # First run - use balanced approach
        strategy["recommended_models"]["red_team"] = current_config.get("red_team_models", [])[:3]
        strategy["recommended_models"]["blue_team"] = current_config.get("blue_team_models", [])[:3]
        strategy["iteration_adjustments"] = {"min_iter": 3, "max_iter": 10}
        return strategy

    # Analyze recent results
    recent_results = results_history[-3:]  # Last 3 iterations
    avg_confidence = sum(r.get("approval_check", {}).get("approval_rate", 0) for r in recent_results) / len(
        recent_results)
    avg_issue_count = sum(len(r.get("agg_risk", {}).get("issues", [])) for r in recent_results) / len(recent_results)

    # Adjust based on performance
    if avg_confidence > 90:
        # High confidence - focus on efficiency
        strategy["recommended_models"]["red_team"] = current_config.get("red_team_models", [])[:2]
        strategy["recommended_models"]["blue_team"] = current_config.get("blue_team_models", [])[:2]
        strategy["iteration_adjustments"] = {"min_iter": 2, "max_iter": 8}
        strategy["focus_areas"] = ["efficiency", "cost_reduction"]
    elif avg_confidence < 70:
        # Low confidence - increase intensity
        strategy["recommended_models"]["red_team"] = current_config.get("red_team_models", [])[:5]
        strategy["recommended_models"]["blue_team"] = current_config.get("blue_team_models", [])[:5]
        strategy["iteration_adjustments"] = {"min_iter": 5, "max_iter": 15}
        strategy["confidence_threshold"] = min(95, strategy["confidence_threshold"] + 5)
        strategy["focus_areas"] = ["thoroughness", "coverage"]
    else:
        # Balanced approach
        strategy["recommended_models"]["red_team"] = current_config.get("red_team_models", [])[:3]
        strategy["recommended_models"]["blue_team"] = current_config.get("blue_team_models", [])[:3]
        strategy["iteration_adjustments"] = {"min_iter": 3, "max_iter": 12}
        strategy["focus_areas"] = ["balanced_approach"]

    return strategy


def category_focused_testing(issues_by_category: Dict[str, int], current_config: Dict) -> Dict[str, Any]:
    """Focus testing on specific issue categories.

    Args:
        issues_by_category (Dict[str, int]): Count of issues by category
        current_config (Dict): Current testing configuration

    Returns:
        Dict[str, Any]: Category-focused testing recommendations
    """
    if not issues_by_category:
        return {"focus_category": None, "recommended_models": {"red_team": [], "blue_team": []}}

    # Find category with most issues
    focus_category = max(issues_by_category.items(), key=lambda x: x[1])[0]

    # Recommend models based on category
    category_experts = {
        "security": ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-1.5-pro"],
        "compliance": ["openai/gpt-4o", "mistral/mistral-medium-latest"],
        "clarity": ["openai/gpt-4o-mini", "google/gemini-1.5-flash"],
        "completeness": ["anthropic/claude-3-sonnet", "meta-llama/llama-3-70b-instruct"],
        "efficiency": ["openai/gpt-4o", "meta-llama/llama-3-70b-instruct"]
    }

    recommended_models = category_experts.get(focus_category, current_config.get("red_team_models", [])[:3])

    return {
        "focus_category": focus_category,
        "recommended_models": {
            "red_team": recommended_models,
            "blue_team": current_config.get("blue_team_models", [])[:3]
        }
    }


def performance_based_model_rotation(model_performance: Dict[str, Dict],
                                     current_red_team: List[str],
                                     current_blue_team: List[str]) -> Dict[str, List[str]]:
    """Rotate models based on performance metrics.

    Args:
        model_performance (Dict[str, Dict]): Performance data for each model
        current_red_team (List[str]): Current red team models
        current_blue_team (List[str]): Current blue team models

    Returns:
        Dict[str, List[str]]: Updated model selections
    """
    # Sort models by performance score
    sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get("score", 0), reverse=True)

    # Select top performers for red team (critics)
    top_red_models = [model_id for model_id, _ in sorted_models[:3] if model_id in current_red_team]
    if not top_red_models:
        top_red_models = current_red_team[:min(3, len(current_red_team))]

    # Select diverse models for blue team (fixers)
    top_blue_models = [model_id for model_id, _ in sorted_models[:3] if model_id in current_blue_team]
    if not top_blue_models:
        top_blue_models = current_blue_team[:min(3, len(current_blue_team))]

    return {
        "red_team": top_red_models,
        "blue_team": top_blue_models
    }


# ------------------------------------------------------------------
# Advanced Analytics Functions
# ------------------------------------------------------------------

def analyze_code_quality(code_text: str) -> Dict[str, Any]:
    """Analyze code quality metrics.

    Args:
        code_text (str): The code to analyze

    Returns:
        Dict[str, Any]: Code quality metrics
    """
    if not code_text:
        return {
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0,
            "comments": 0,
            "complexity": 0,
            "duplicate_lines": 0,
            "quality_score": 0
        }

    lines = code_text.split('\n')
    lines_of_code = len([line for line in lines if line.strip()])

    # Count functions (simplified)
    function_patterns = [r'\bdef\s+\w+\s*\(', r'\bfunction\s+\w+\s*\(', r'\w+\s*\([^)]*\)\s*{']
    functions = 0
    for pattern in function_patterns:
        functions += len(re.findall(pattern, code_text))

    # Count classes (simplified)
    class_patterns = [r'\bclass\s+\w+', r'\bstruct\s+\w+']
    classes = 0
    for pattern in class_patterns:
        classes += len(re.findall(pattern, code_text))

    # Count comments (simplified)
    comment_patterns = [r'#.*', r'//.*', r'/\*.*?\*/', r'<!--.*?-->']
    comments = 0
    for pattern in comment_patterns:
        comments += len(re.findall(pattern, code_text, re.DOTALL))

    # Simplified complexity calculation
    complexity_keywords = ['if', 'for', 'while', 'switch', 'try', 'catch', '&&', '||']
    complexity = 0
    for keyword in complexity_keywords:
        complexity += code_text.lower().count(keyword)

    # Estimate duplicate lines (very simplified)
    unique_lines = len(set(lines))
    duplicate_lines = lines_of_code - unique_lines if lines_of_code > 0 else 0

    # Calculate quality score (simplified)
    quality_score = 100
    if lines_of_code > 0:
        # Deduct points for low comment ratio
        comment_ratio = comments / lines_of_code if lines_of_code > 0 else 0
        if comment_ratio < 0.1:
            quality_score -= (0.1 - comment_ratio) * 100 * 2  # Up to 20 points deduction

        # Deduct points for high complexity
        complexity_ratio = complexity / lines_of_code if lines_of_code > 0 else 0
        if complexity_ratio > 0.3:
            quality_score -= (complexity_ratio - 0.3) * 100 * 1.5  # Up to 15 points deduction

        # Deduct points for duplicate lines
        duplicate_ratio = duplicate_lines / lines_of_code if lines_of_code > 0 else 0
        if duplicate_ratio > 0.05:
            quality_score -= (duplicate_ratio - 0.05) * 100 * 3  # Up to 30 points deduction

    quality_score = max(0, min(100, quality_score))  # Clamp to 0-100

    return {
        "lines_of_code": lines_of_code,
        "functions": functions,
        "classes": classes,
        "comments": comments,
        "complexity": complexity,
        "duplicate_lines": duplicate_lines,
        "quality_score": round(quality_score, 2)
    }


def analyze_plan_quality(plan_text: str) -> Dict[str, Any]:
    """Analyze plan quality metrics.

    Args:
        plan_text (str): The plan to analyze

    Returns:
        Dict[str, Any]: Plan quality metrics
    """
    if not plan_text:
        return {
            "sections": 0,
            "objectives": 0,
            "milestones": 0,
            "resources": 0,
            "risks": 0,
            "dependencies": 0,
            "timeline_elements": 0,
            "quality_score": 0
        }

    # Count sections (headers)
    sections = len(re.findall(r'^#{1,6}\s+|.*\n[=]{3,}|.*\n[-]{3,}', plan_text, re.MULTILINE))

    # Count objectives (look for objective-related terms)
    objective_patterns = [r'\bobjectives?\b', r'\bgoals?\b', r'\bpurpose\b', r'\baim\b']
    objectives = 0
    for pattern in objective_patterns:
        objectives += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count milestones (look for milestone-related terms)
    milestone_patterns = [r'\bmilestones?\b', r'\bdeadlines?\b', r'\btimelines?\b', r'\bschedule\b']
    milestones = 0
    for pattern in milestone_patterns:
        milestones += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count resources (look for resource-related terms)
    resource_patterns = [r'\bresources?\b', r'\bbudget\b', r'\bcosts?\b', r'\bmaterials?\b']
    resources = 0
    for pattern in resource_patterns:
        resources += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count risks (look for risk-related terms)
    risk_patterns = [r'\brisks?\b', r'\bthreats?\b', r'\bvulnerabilit(?:y|ies)\b', r'\bhazards?\b']
    risks = 0
    for pattern in risk_patterns:
        risks += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count dependencies (look for dependency-related terms)
    dependency_patterns = [r'\bdependenc(?:y|ies)\b', r'\bprerequisites?\b', r'\brequires?\b', r'\bneeds?\b']
    dependencies = 0
    for pattern in dependency_patterns:
        dependencies += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count timeline elements (dates, time-related terms)
    timeline_patterns = [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                         r'\bweeks?\b', r'\bmonths?\b', r'\byears?\b', r'\bdays?\b']
    timeline_elements = 0
    for pattern in timeline_patterns:
        timeline_elements += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Calculate quality score (simplified)
    quality_score = 50  # Start with baseline

    # Add points for completeness
    quality_score += min(20, sections * 2)  # Up to 20 points for sections
    quality_score += min(15, objectives * 3)  # Up to 15 points for objectives
    quality_score += min(10, milestones * 2)  # Up to 10 points for milestones
    quality_score += min(10, resources * 2)  # Up to 10 points for resources
    quality_score += min(10, risks * 2)  # Up to 10 points for risks
    quality_score += min(10, dependencies * 2)  # Up to 10 points for dependencies
    quality_score += min(10, timeline_elements * 2)  # Up to 10 points for timeline

    quality_score = max(0, min(100, quality_score))  # Clamp to 0-100

    return {
        "sections": sections,
        "objectives": objectives,
        "milestones": milestones,
        "resources": resources,
        "risks": risks,
        "dependencies": dependencies,
        "timeline_elements": timeline_elements,
        "quality_score": round(quality_score, 2)
    }


def run_adversarial_testing():
    """Main logic for the adversarial testing loop, designed to be run in a background thread."""
    try:
        # --- Initialization ---
        api_key = st.session_state.openrouter_key
        red_team_base = list(st.session_state.red_team_models or [])
        blue_team_base = list(st.session_state.blue_team_models or [])
        min_iter, max_iter = st.session_state.adversarial_min_iter, st.session_state.adversarial_max_iter
        confidence = st.session_state.adversarial_confidence
        max_tokens = st.session_state.adversarial_max_tokens
        json_mode = st.session_state.adversarial_force_json
        max_workers = st.session_state.adversarial_max_workers
        rotation_strategy = st.session_state.adversarial_rotation_strategy
        seed_str = str(st.session_state.adversarial_seed or "").strip()
        seed = None
        if seed_str:
            try:
                seed = int(float(seed_str))  # Handle floats by truncating to int
            except (ValueError, TypeError):
                pass  # Invalid input, keep seed as None

        # Validation
        if not api_key:
            _update_adv_log_and_status("❌ Error: OpenRouter API key is required for adversarial testing.")
            with st.session_state.thread_lock:
                st.session_state.adversarial_running = False
            return

        if not red_team_base or not blue_team_base:
            _update_adv_log_and_status("❌ Error: Please select at least one model for both red and blue teams.")
            with st.session_state.thread_lock:
                st.session_state.adversarial_running = False
            return

        if not st.session_state.protocol_text.strip():
            _update_adv_log_and_status("❌ Error: Please enter a protocol to test.")
            with st.session_state.thread_lock:
                st.session_state.adversarial_running = False
            return

        with st.session_state.thread_lock:
            st.session_state.adversarial_log = []
            st.session_state.adversarial_stop_flag = False
            st.session_state.adversarial_total_tokens_prompt = 0
            st.session_state.adversarial_total_tokens_completion = 0
            st.session_state.adversarial_cost_estimate_usd = 0.0

        model_configs = _collect_model_configs(red_team_base + blue_team_base, max_tokens)
        current_sop = st.session_state.protocol_text
        base_hash = _hash_text(current_sop)
        results, sop_hashes = [], [base_hash]
        iteration, approval_rate = 0, 0.0

        # Determine review type and get appropriate prompts
        if st.session_state.get("adversarial_custom_mode", False):
            # Use custom prompts when custom mode is enabled
            red_team_prompt = st.session_state.get("adversarial_custom_red_prompt", RED_TEAM_CRITIQUE_PROMPT)
            blue_team_prompt = st.session_state.get("adversarial_custom_blue_prompt", BLUE_TEAM_PATCH_PROMPT)
            review_type = "custom"
        else:
            # Use standard prompts based on review type
            if st.session_state.adversarial_review_type == "Auto-Detect":
                review_type = determine_review_type(current_sop)
            elif st.session_state.adversarial_review_type == "Code Review":
                review_type = "code"
            elif st.session_state.adversarial_review_type == "Plan Review":
                review_type = "plan"
            else:
                review_type = "general"

            red_team_prompt, blue_team_prompt = get_appropriate_prompts(review_type)

        _update_adv_log_and_status(
            f"🚀 Start: {len(red_team_base)} red / {len(blue_team_base)} blue | seed={seed} | base_hash={base_hash} | rotation={rotation_strategy} | review_type={review_type}")

        # --- Main Loop ---
        while iteration < max_iter and not st.session_state.adversarial_stop_flag:
            iteration += 1

            # --- Team Rotation Logic ---
            if rotation_strategy == "Round Robin":
                red_team = [red_team_base[(iteration - 1 + i) % len(red_team_base)] for i in range(len(red_team_base))]
                blue_team = [blue_team_base[(iteration - 1 + i) % len(blue_team_base)] for i in
                             range(len(blue_team_base))]
                _update_adv_log_and_status(
                    f"🔄 Iteration {iteration}/{max_iter}: Rotated teams (Round Robin). Red: {len(red_team)}, Blue: {len(blue_team)}")
            elif rotation_strategy == "Staged":
                try:
                    stages = json.loads(st.session_state.adversarial_staged_rotation_config)
                    if isinstance(stages, list) and len(stages) > 0:
                        stage_index = (iteration - 1) % len(stages)
                        stage = stages[stage_index]
                        red_team = stage.get("red", red_team_base)
                        blue_team = stage.get("blue", blue_team_base)
                        _update_adv_log_and_status(
                            f"🔄 Iteration {iteration}/{max_iter}: Rotated teams (Staged - Stage {stage_index + 1}). Red: {len(red_team)}, Blue: {len(blue_team)}")
                    else:
                        red_team = red_team_base
                        blue_team = blue_team_base
                        _update_adv_log_and_status(f"⚠️ Invalid Staged Rotation Config. Using base teams.")
                except json.JSONDecodeError:
                    red_team = red_team_base
                    blue_team = blue_team_base
                    _update_adv_log_and_status(f"⚠️ Invalid JSON in Staged Rotation Config. Using base teams.")
            elif rotation_strategy == "Performance-Based":
                model_performance = st.session_state.adversarial_model_performance
                red_team_weights = [model_performance.get(m, {"score": 1})["score"] for m in red_team_base]
                red_team_sample_size = min(st.session_state.adversarial_red_team_sample_size, len(red_team_base))
                if sum(red_team_weights) == 0:
                    red_team = random.sample(red_team_base, k=red_team_sample_size)
                else:
                    red_team = random.choices(red_team_base, weights=red_team_weights, k=red_team_sample_size)

                blue_team_sample_size = min(st.session_state.adversarial_blue_team_sample_size, len(blue_team_base))
                blue_team = random.sample(blue_team_base, k=blue_team_sample_size)
                _update_adv_log_and_status(
                    f"🔄 Iteration {iteration}/{max_iter}: Rotated teams (Performance-Based). Red: {len(red_team)}, Blue: {len(blue_team)}")
            elif rotation_strategy == "Adaptive":
                # Adaptive strategy based on previous iteration performance
                if iteration > 1 and len(results) > 0:
                    last_iteration = results[-1]
                    # If approval rate is low, use more diverse models
                    if last_iteration.get("approval_check", {}).get("approval_rate", 100) < 70:
                        red_team = random.sample(red_team_base, min(len(red_team_base),
                                                                    st.session_state.adversarial_red_team_sample_size + 1))
                        blue_team = random.sample(blue_team_base, min(len(blue_team_base),
                                                                      st.session_state.adversarial_blue_team_sample_size + 1))
                        _update_adv_log_and_status(
                            f"🔄 Iteration {iteration}/{max_iter}: Adaptive rotation - increasing diversity. Red: {len(red_team)}, Blue: {len(blue_team)}")
                    # If approval rate is high, focus on specialized models
                    elif last_iteration.get("approval_check", {}).get("approval_rate", 0) > 90:
                        # Use top performing models
                        top_red_models = sorted(st.session_state.adversarial_model_performance.items(),
                                                key=lambda x: x[1].get("score", 0), reverse=True)[:3]
                        top_red_model_ids = [m[0] for m in top_red_models if m[0] in red_team_base]

                        top_blue_models = sorted(st.session_state.adversarial_model_performance.items(),
                                                 key=lambda x: x[1].get("score", 0), reverse=True)[:3]
                        top_blue_model_ids = [m[0] for m in top_blue_models if m[0] in blue_team_base]

                        if top_red_model_ids:
                            red_team = top_red_model_ids
                        else:
                            red_team = red_team_base[:min(3, len(red_team_base))]

                        if top_blue_model_ids:
                            blue_team = top_blue_model_ids
                        else:
                            blue_team = blue_team_base[:min(3, len(blue_team_base))]
                        _update_adv_log_and_status(
                            f"🔄 Iteration {iteration}/{max_iter}: Adaptive rotation - focusing on top models. Red: {len(red_team)}, Blue: {len(blue_team)}")
                    else:
                        red_team = red_team_base
                        blue_team = blue_team_base
                else:
                    red_team = red_team_base
                    blue_team = blue_team_base
                _update_adv_log_and_status(
                    f"🔄 Iteration {iteration}/{max_iter}: Adaptive team selection. Red: {len(red_team)}, Blue: {len(blue_team)}")

            # Advanced Testing Strategies
            elif "Adaptive Testing" in st.session_state.get("advanced_testing_strategies", []):
                # Use adaptive testing strategy
                strategy = adaptive_testing_strategy(results, {
                    "red_team_models": red_team_base,
                    "blue_team_models": blue_team_base,
                    "adversarial_confidence": confidence
                })
                red_team = strategy["recommended_models"]["red_team"]
                blue_team = strategy["recommended_models"]["blue_team"]
                _update_adv_log_and_status(
                    f"🔄 Iteration {iteration}/{max_iter}: Adaptive testing strategy applied. Red: {len(red_team)}, Blue: {len(blue_team)}")

            elif "Category-Focused Testing" in st.session_state.get("advanced_testing_strategies", []):
                # Focus on specific issue categories
                if results and "agg_risk" in results[-1]:
                    categories = results[-1]["agg_risk"].get("categories", {})
                    if categories:
                        focus_recommendation = category_focused_testing(categories, {
                            "red_team_models": red_team_base,
                            "blue_team_models": blue_team_base
                        })
                        red_team = focus_recommendation["recommended_models"]["red_team"]
                        blue_team = focus_recommendation["recommended_models"]["blue_team"]
                        focus_category = focus_recommendation["focus_category"]
                        _update_adv_log_and_status(
                            f"🔄 Iteration {iteration}/{max_iter}: Category-focused testing on '{focus_category}'. Red: {len(red_team)}, Blue: {len(blue_team)}")
                    else:
                        red_team = red_team_base
                        blue_team = blue_team_base
                else:
                    red_team = red_team_base
                    blue_team = blue_team_base

            elif "Performance-Based Rotation" in st.session_state.get("advanced_testing_strategies", []):
                # Rotate models based on performance
                if st.session_state.get("adversarial_model_performance"):
                    rotated_teams = performance_based_model_rotation(
                        st.session_state.adversarial_model_performance,
                        red_team_base,
                        blue_team_base
                    )
                    red_team = rotated_teams["red_team"]
                    blue_team = rotated_teams["blue_team"]
                    _update_adv_log_and_status(
                        f"🔄 Iteration {iteration}/{max_iter}: Performance-based rotation. Red: {len(red_team)}, Blue: {len(blue_team)}")
                else:
                    red_team = red_team_base
                    blue_team = blue_team_base

            else:  # "None" or any other case
                red_team = red_team_base
                blue_team = blue_team_base

            _update_adv_log_and_status(f"🔄 Iteration {iteration}/{max_iter}: Starting red team analysis.")

            # --- RED TEAM: CRITIQUES ---
            critiques_raw = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(analyze_with_model, api_key, m, current_sop,
                                     model_configs.get(m, {}), red_team_prompt,
                                     force_json=json_mode, seed=seed,
                                     compliance_requirements=st.session_state.compliance_requirements): m for m in
                           red_team}
                for fut in as_completed(futures):
                    res = fut.result()
                    _update_adv_counters(res['ptoks'], res['ctoks'], res['cost'])
                    if not res.get("ok") or not res.get("json"):
                        _update_adv_log_and_status(
                            f"🔴 {res['model_id']}: Invalid response. Details: {res.get('text', 'N/A')}")
                    critiques_raw.append(
                        {"model": res['model_id'], "critique_json": res.get("json"), "raw_text": res.get("text")})

            _update_model_performance(critiques_raw)
            agg_risk = _aggregate_red_risk(critiques_raw)
            if agg_risk['count'] == 0:
                _update_adv_log_and_status(
                    f"🔄 Iteration {iteration}: Red team found no exploitable issues. Checking for approval.")
            else:
                _update_adv_log_and_status(
                    f"🔄 Iteration {iteration}: Red team found {agg_risk['count']} issues. Starting blue team patching.")

            # --- BLUE TEAM: PATCHING ---
            blue_patches_raw = []
            valid_critiques_json = [c['critique_json'] for c in critiques_raw if c.get('critique_json')]
            critique_block = json.dumps({"critiques": valid_critiques_json}, ensure_ascii=False, indent=2)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(analyze_with_model, api_key, m, current_sop,
                                     model_configs.get(m, {}), blue_team_prompt,
                                     user_suffix="\n\nCRITIQUES TO ADDRESS:\n" + critique_block,
                                     force_json=True, seed=seed): m for m in blue_team}
                for fut in as_completed(futures):
                    res = fut.result()
                    _update_adv_counters(res['ptoks'], res['ctoks'], res['cost'])
                    if not res.get("ok") or not res.get("json") or not res.get("json", {}).get("sop", "").strip():
                        _update_adv_log_and_status(
                            f"🔵 {res['model_id']}: Invalid or empty patch received. Details: {res.get('text', 'N/A')}")
                    blue_patches_raw.append(
                        {"model": res['model_id'], "patch_json": res.get("json"), "raw_text": res.get("text")})

            next_sop, consensus_diag = _merge_consensus_sop(current_sop, blue_patches_raw, critiques_raw)
            _update_adv_log_and_status(
                f"🔄 Iteration {iteration}: Consensus SOP generated (Best patch from '{consensus_diag.get('model', 'N/A')}'). Starting approval check.")

            # --- APPROVAL CHECK ---
            # Use custom approval prompt when in custom mode
            if st.session_state.get("adversarial_custom_mode", False):
                approval_prompt = st.session_state.get("adversarial_custom_approval_prompt", APPROVAL_PROMPT)
            else:
                approval_prompt = APPROVAL_PROMPT

            eval_res = check_approval_rate(api_key, red_team, next_sop, model_configs, seed, max_workers,
                                           approval_prompt)
            approval_rate = eval_res["approval_rate"]
            _update_adv_counters(eval_res['prompt_tokens'], eval_res['completion_tokens'], eval_res['cost'])
            _update_adv_log_and_status(
                f"🔄 Iteration {iteration}: Approval rate: {approval_rate:.1f}%, Avg Score: {eval_res['avg_score']:.1f}")

            results.append({
                "iteration": iteration, "critiques": critiques_raw, "patches": blue_patches_raw,
                "current_sop": next_sop, "approval_check": eval_res, "agg_risk": agg_risk, "consensus": consensus_diag,
                "cost_effectiveness": (agg_risk[
                                           'count'] / st.session_state.adversarial_cost_estimate_usd) if st.session_state.adversarial_cost_estimate_usd > 0 else 0
            })
            current_sop = next_sop

            # --- Confidence Plateau and Critical Issue Triggers ---
            with st.session_state.thread_lock:
                st.session_state.adversarial_confidence_history.append(approval_rate)
                history = st.session_state.adversarial_confidence_history
                if len(history) > 3 and history[-1] == history[-2] and history[-2] == history[-3]:
                    _update_adv_log_and_status(
                        "⚠️ Confidence plateau detected: Confidence has not changed for 3 iterations.")

            if agg_risk["total_weight"] > 0 and any(
                    issue.get("severity") == "critical" for critique in critiques_raw if critique.get("critique_json")
                    for issue in critique["critique_json"].get("issues", [])):
                _update_adv_log_and_status(
                    "🚨 Critical issue found! Activating specialist security models (simulation).")

            # --- Stagnation Check ---
            current_hash = _hash_text(current_sop)
            if len(sop_hashes) > 1 and current_hash == sop_hashes[-1] and current_hash == sop_hashes[-2]:
                _update_adv_log_and_status(
                    "⚠️ Stagnation detected: SOP has not changed for 2 iterations. Consider adjusting models or temperature.")
            sop_hashes.append(current_hash)

            if iteration >= min_iter and approval_rate >= confidence:
                _update_adv_log_and_status(
                    f"✅ Success! Confidence threshold of {confidence}% reached after {iteration} iterations.")
                break
        # --- End of Loop ---
        if st.session_state.adversarial_stop_flag:
            _update_adv_log_and_status("⏹️ Process stopped by user.")
        elif iteration >= max_iter:
            _update_adv_log_and_status(
                f"🏁 Reached max iterations ({max_iter}). Final approval rate: {approval_rate:.1f}%")

        with st.session_state.thread_lock:
            st.session_state.adversarial_results = {
                "final_sop": current_sop, "iterations": results, "final_approval_rate": approval_rate,
                "cost_estimate_usd": st.session_state.adversarial_cost_estimate_usd,
                "tokens": {"prompt": st.session_state.adversarial_total_tokens_prompt,
                           "completion": st.session_state.adversarial_total_tokens_completion},
                "log": list(st.session_state.adversarial_log), "seed": seed, "base_hash": base_hash,
                "review_type": review_type
            }
            st.session_state.protocol_text = current_sop
            st.session_state.adversarial_running = False

    except Exception as e:
        # --- Global Error Handler ---
        tb_str = traceback.format_exc()
        error_message = f"💥 A critical error occurred: {e}\n{tb_str}"
        _update_adv_log_and_status(error_message)
        with st.session_state.thread_lock:
            st.session_state.adversarial_running = False
            if 'adversarial_results' not in st.session_state or not st.session_state.adversarial_results:
                st.session_state.adversarial_results = {}
            st.session_state.adversarial_results["critical_error"] = error_message
            # Ensure error is visible in UI by storing a simplified message
            st.session_state.adversarial_status_message = f"Error: {str(e)[:100]}..."
