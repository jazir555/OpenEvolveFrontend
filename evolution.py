# ------------------------------------------------------------------
# 8. Run logic for Evolution tab (Self-Contained Implementation)
# ------------------------------------------------------------------

def _request_openai_compatible_chat(
        api_key: str, base_url: str, model: str, messages: List, extra_headers: Dict,
        temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float,
        max_tokens: int, seed: Optional[int], req_timeout: int = 60, max_retries: int = 5,
        provider: str = "OpenAI"
) -> str:
    url = base_url.rstrip('/') + "/chat/completions"
    headers = {"Content-Type": "application/json", **extra_headers}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
        "top_p": top_p, "frequency_penalty": frequency_penalty, "presence_penalty": presence_penalty,
    }
    if PROVIDERS.get(provider, {}).get("omit_model_in_payload"):
        payload.pop("model", None)
    if seed is not None:
        payload["seed"] = int(seed)

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)

            # Handle rate limiting and server errors with retry
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"HTTP {r.status_code}: {r.text}")
                continue

            r.raise_for_status()

            # Handle non-JSON responses
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                last_err = Exception(f"Invalid JSON response: {e} - Response: {r.text[:200]}...")
                time.sleep((2 ** attempt) + _rand_jitter_ms())
                continue

            # Safely access the response structure
            if not isinstance(data, dict):
                last_err = Exception(f"Unexpected response format: {type(data)} - Response: {data}")
                time.sleep((2 ** attempt) + _rand_jitter_ms())
                continue

            # Handle API-specific error formats
            if "error" in data:
                error_msg = data["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                last_err = Exception(f"API error: {error_msg}")
                time.sleep((2 ** attempt) + _rand_jitter_ms())
                continue

            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
                if content is not None:
                    return content
                else:
                    last_err = Exception("Empty content in response choice")
            else:
                last_err = Exception("No choices in response")

            # If we get here, there was an issue with the response structure
            time.sleep((2 ** attempt) + _rand_jitter_ms())

        except requests.exceptions.ConnectionError as e:
            last_err = Exception(f"Connection error: {e}")
            time.sleep((2 ** attempt) + _rand_jitter_ms())
        except requests.exceptions.Timeout as e:
            last_err = Exception(f"Request timeout: {e}")
            time.sleep((2 ** attempt) + _rand_jitter_ms())
        except requests.exceptions.RequestException as e:
            last_err = Exception(f"Request failed: {e}")
            time.sleep((2 ** attempt) + _rand_jitter_ms())
        except Exception as e:
            last_err = e
            time.sleep((2 ** attempt) + _rand_jitter_ms())

    raise RuntimeError(f"Request failed after {max_retries} attempts for model {model}: {last_err}")


def run_evolution_internal():
    try:
        with st.session_state.thread_lock:
            st.session_state.evolution_log = []
            st.session_state.evolution_stop_flag = False
        current_protocol = st.session_state.protocol_text
        st.session_state.evolution_current_best = current_protocol

        def log_msg(msg):
            with st.session_state.thread_lock:
                st.session_state.evolution_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        log_msg(f"🚀 Starting evolution process with {st.session_state.provider}/{st.session_state.model}...")
        try:
            extra_hdrs = json.loads(st.session_state.extra_headers or "{}")
            if not isinstance(extra_hdrs, dict): raise json.JSONDecodeError("JSON is not a dictionary.", "", 0)
        except (ValueError, TypeError):
            log_msg("⚠️ Invalid Extra Headers JSON. Must be a dictionary. Using empty dict.")
            extra_hdrs = {}

        seed_str = str(st.session_state.seed or "").strip()
        seed = None
        if seed_str:
            try:
                seed = int(float(seed_str))  # Handle floats by truncating to int
            except (ValueError, TypeError):
                pass  # Invalid input, keep seed as None

        api_key_to_use = st.session_state.api_key
        provider_info = PROVIDERS.get(st.session_state.provider, {})
        if not api_key_to_use and (env_var := provider_info.get("env")):
            api_key_to_use = os.environ.get(env_var, "")
            if api_key_to_use: log_msg(f"Using API key from env var {env_var}.")

        if not api_key_to_use and provider_info.get("env"):
            log_msg(f"⚠️ No API key provided in UI or in env var {provider_info.get('env')}. Requests may fail.")

        request_functions = {
            "Anthropic": _request_anthropic_chat,
            "Google (Gemini)": _request_google_gemini_chat,
            "Cohere": _request_cohere_chat,
        }
        request_function = request_functions.get(st.session_state.provider, _request_openai_compatible_chat)

        consecutive_failures = 0
        max_consecutive_failures = 3  # Stop after 3 consecutive failures

        # Initialize metrics tracking
        iteration_metrics = []

        for i in range(st.session_state.max_iterations):
            if st.session_state.evolution_stop_flag:
                log_msg("⏹️ Evolution stopped by user.")
                break
            log_msg(f"🔄 --- Iteration {i + 1}/{st.session_state.max_iterations} ---")
            try:
                messages = _compose_messages(
                    st.session_state.system_prompt,
                    f"Current draft:\n\n---\n{current_protocol}\n---\n\nImprove it based on your instructions."
                )

                kwargs = {
                    "api_key": api_key_to_use,
                    "base_url": st.session_state.base_url,
                    "model": st.session_state.model,
                    "messages": messages,
                    "extra_headers": extra_hdrs,
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "max_tokens": st.session_state.max_tokens,
                    "seed": seed,
                }
                if st.session_state.provider not in request_functions:
                    kwargs["frequency_penalty"] = st.session_state.frequency_penalty
                    kwargs["presence_penalty"] = st.session_state.presence_penalty
                    kwargs["provider"] = st.session_state.provider

                improved_protocol = request_function(**kwargs)

                # Enhanced validation with quality metrics
                if improved_protocol and len(improved_protocol.strip()) > len(current_protocol) * 0.7:
                    # Calculate quality metrics
                    original_complexity = calculate_protocol_complexity(current_protocol)
                    improved_complexity = calculate_protocol_complexity(improved_protocol)

                    # Quality improvement check
                    quality_improvement = (
                            improved_complexity["complexity_score"] >= original_complexity["complexity_score"] and
                            improved_complexity["unique_words"] >= original_complexity["unique_words"] * 0.9
                    )

                    if quality_improvement or len(improved_protocol.strip()) > len(current_protocol) * 1.1:
                        log_msg(f"✅ Iteration {i + 1} successful. Length: {len(improved_protocol.strip())}")
                        current_protocol = improved_protocol.strip()
                        st.session_state.evolution_current_best = current_protocol
                        consecutive_failures = 0  # Reset failure counter on success

                        # Track metrics
                        iteration_metrics.append({
                            "iteration": i + 1,
                            "length": len(current_protocol),
                            "complexity": improved_complexity["complexity_score"],
                            "unique_words": improved_complexity["unique_words"],
                            "improvement": len(current_protocol) - len(st.session_state.protocol_text)
                        })
                    else:
                        log_msg(
                            f"⚠️ Iteration {i + 1} result rejected (no significant quality improvement). Length: {len(improved_protocol.strip() if improved_protocol else '')}")
                        consecutive_failures += 1
                else:
                    log_msg(
                        f"⚠️ Iteration {i + 1} result rejected (too short or empty). Length: {len(improved_protocol.strip() if improved_protocol else '')}")
                    consecutive_failures += 1
                if (i + 1) % st.session_state.checkpoint_interval == 0:
                    log_msg(f"💾 Checkpoint at iteration {i + 1}.")
            except Exception as e:
                log_msg(f"❌ ERROR in iteration {i + 1}: {e}")
                consecutive_failures += 1
                time.sleep(2)

            # Stop if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                log_msg(f"🛑 Stopping evolution due to {consecutive_failures} consecutive failures.")
                break

        log_msg("🏁 Evolution finished.")

        # Log final metrics
        if iteration_metrics:
            final_length = iteration_metrics[-1]["length"]
            initial_length = len(st.session_state.protocol_text)
            total_improvement = final_length - initial_length
            log_msg(
                f"📊 Total improvement: {total_improvement} characters ({(total_improvement / initial_length) * 100:.1f}%)")

        st.session_state.protocol_text = current_protocol  # Update the main protocol text
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_msg(f"💥 A critical error occurred in the evolution thread: {e}\n{tb_str}")
    finally:
        st.session_state.evolution_running = False



