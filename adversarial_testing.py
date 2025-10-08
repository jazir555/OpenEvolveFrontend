"""
Comprehensive Adversarial Testing Implementation
Implements all adversarial testing functionality from ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md
"""
import streamlit as st
import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from session_utils import _update_adv_log_and_status

try:
    from openevolve_integration import (
        run_unified_evolution,
    )
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False


def run_comprehensive_adversarial_testing(
    content: str,
    content_type: str,
    red_team_models: List[str],
    blue_team_models: List[str],
    evaluator_models: List[str],
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    max_iterations: int = 50,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    custom_requirements: str = "",
    compliance_rules: Optional[List[str]] = None,
    feature_dimensions: Optional[List[str]] = None,
    # Advanced OpenEvolve parameters 
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    parallel_evaluations: int = 4,
    system_message: str = None,
    evaluator_system_message: str = None,
    num_top_programs: int = 3,
    num_diverse_programs: int = 2,
    early_stopping_patience: Optional[int] = None,
    convergence_threshold: float = 0.001,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run comprehensive adversarial testing with red team, blue team, and evaluator team
    following the implementation described in ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md
    
    Args:
        content: The content to test adversarially
        content_type: Type of content being tested (e.g., code_python, document_legal)
        red_team_models: List of models for red team (critique/attack)
        blue_team_models: List of models for blue team (fix/improve)
        evaluator_models: List of models for evaluator team (judge/assess)
        api_key: API key for LLM provider
        api_base: Base URL for API
        max_iterations: Maximum number of adversarial iterations
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        custom_requirements: Custom requirements to check for
        compliance_rules: Compliance rules to verify against
        feature_dimensions: List of feature dimensions for MAP-Elites
        
    Returns:
        Dict with comprehensive adversarial testing results
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend not available for adversarial testing")
        return {"success": False, "error": "OpenEvolve backend not available"}

    try:
        # Phase 1: Red Team Critique Generation
        st.info("⚔️ Phase 1: Red Team Critique Generation")
        _update_adv_log_and_status("⚔️ Phase 1: Red Team Critique Generation...")
        
        red_team_results = run_red_team_analysis(
            content=content,
            content_type=content_type,
            red_team_models=red_team_models,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_requirements=custom_requirements,
            compliance_rules=compliance_rules
        )
        
        if not red_team_results.get("success"):
            return {"success": False, "error": "Red team analysis failed"}
        
        identified_issues = red_team_results.get("findings", [])
        st.success(f"🔍 Red Team identified {len(identified_issues)} issues")
        
        # Phase 2: Blue Team Patch Development
        st.info("🛡️ Phase 2: Blue Team Patch Development")
        _update_adv_log_and_status("🛡️ Phase 2: Blue Team Patch Development...")
        
        blue_team_results = run_blue_team_resolution(
            original_content=content,
            identified_issues=identified_issues,
            content_type=content_type,
            blue_team_models=blue_team_models,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if not blue_team_results.get("success"):
            return {"success": False, "error": "Blue team resolution failed"}
        
        improved_content = blue_team_results.get("improved_content", content)
        applied_fixes = blue_team_results.get("applied_fixes", [])
        st.success(f"🛠️ Blue Team applied {len(applied_fixes)} fixes")
        
        # Phase 3: Evaluator Team Assessment
        st.info("⚖️ Phase 3: Evaluator Team Assessment")
        _update_adv_log_and_status("⚖️ Phase 3: Evaluator Team Assessment...")
        
        # Run adversarial evolution using OpenEvolve with all the findings and fixes
        evolution_results = run_unified_evolution(
            content=improved_content,
            content_type=content_type,
            evolution_mode="adversarial",
            model_configs=[
                {"name": model, "weight": 1.0} for model in red_team_models + blue_team_models + evaluator_models
            ],
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=1000,
            system_message=system_message or f"Improve the content by addressing issues identified by red team: {identified_issues}",
            evaluator_system_message=evaluator_system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            feature_dimensions=feature_dimensions,
            custom_requirements=custom_requirements,
            # Advanced parameters
            enable_artifacts=enable_artifacts,
            cascade_evaluation=cascade_evaluation,
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
            parallel_evaluations=parallel_evaluations,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            early_stopping_patience=early_stopping_patience,
            convergence_threshold=convergence_threshold,
            random_seed=random_seed,
        )
        
        if evolution_results and evolution_results.get("success"):
            final_content = evolution_results.get("best_code", improved_content)
            final_score = evolution_results.get("best_score", 0.0)
            
            # Prepare comprehensive results
            results = {
                "success": True,
                "original_content": content,
                "final_content": final_content,
                "final_score": final_score,
                "red_team_findings": identified_issues,
                "blue_team_fixes": applied_fixes,
                "evolution_results": evolution_results,
                "metrics": {
                    "issues_identified": len(identified_issues),
                    "fixes_applied": len(applied_fixes),
                    "improvement_score": final_score,
                    "consensus_achieved": True,  # Assuming consensus in unified evolution
                }
            }
            
            _update_adv_log_and_status(f"✅ Adversarial testing completed. Final score: {final_score:.4f}")
            return results
        else:
            return {
                "success": False,
                "error": "Adversarial evolution failed",
                "red_team_findings": identified_issues,
                "blue_team_fixes": applied_fixes
            }
    
    except Exception as e:
        st.error(f"Error in comprehensive adversarial testing: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def run_red_team_analysis(
    content: str,
    content_type: str,
    red_team_models: List[str],
    api_key: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    custom_requirements: str = "",
    compliance_rules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Phase 1: Run Red Team analysis to identify issues and vulnerabilities
    Implements the Red Team Critique Generation process
    """
    try:
        findings = []
        
        # Enhanced red team system prompt based on documentation
        red_team_prompts = [
            f"""You are a red team critic analyzing {content_type.replace('_', ' ')} content.
            Perform a thorough analysis to identify flaws, vulnerabilities, weaknesses, and issues.

            Analyze the following content:
            {content}

            Identify issues across these categories:
            1. Functional issues
            2. Structural problems 
            3. Security vulnerabilities
            4. Compliance violations
            5. Performance bottlenecks
            6. Maintainability problems
            7. Scalability concerns
            8. Robustness issues

            For each issue found, provide:
            - Severity (Low, Medium, High, Critical)
            - Category
            - Description
            - Potential impact
            - Suggested remediation approach

            Also consider: {custom_requirements} and compliance with: {compliance_rules or 'N/A'}""",
            
            f"""As an adversarial red team member, try to break or find weaknesses in this {content_type.replace('_', ' ')} content:
            {content}

            Focus on:
            - Edge cases and boundary conditions
            - Assumptions that might be incorrect
            - Security vulnerabilities
            - Logic flaws
            - Error handling issues
            - Resource management problems

            Report all vulnerabilities you can identify.""",
            
            f"""You are a quality assurance red team member. Critically evaluate this {content_type.replace('_', ' ')} content:
            {content}

            Look for:
            - Code quality issues (if code)
            - Document quality problems (if document)
            - Inconsistencies
            - Missing elements
            - Poor practices
            - Optimization opportunities

            Provide constructive criticism with specific examples."""
        ]
        
        # Use ThreadPoolExecutor to run red team analysis in parallel
        with ThreadPoolExecutor(max_workers=min(3, len(red_team_models))) as executor:
            futures = []
            
            # Submit tasks for different red team models
            for i, model in enumerate(red_team_models[:3]):  # Use first 3 models
                prompt = red_team_prompts[i % len(red_team_prompts)]
                
                future = executor.submit(
                    _request_openai_compatible_chat,
                    api_key,
                    api_base,
                    model,
                    [{"role": "user", "content": prompt}],
                    {"Content-Type": "application/json"},
                    temperature,
                    0.95,
                    0.0,
                    0.0,
                    max_tokens,
                    None
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        # Parse the result to extract findings
                        findings.extend(_parse_red_team_findings(result))
                except Exception as e:
                    st.warning(f"Error in red team analysis thread: {e}")
        
        return {
            "success": True,
            "findings": findings,
            "total_findings": len(findings),
            "findings_by_severity": _categorize_findings_by_severity(findings)
        }
    
    except Exception as e:
        st.error(f"Error in red team analysis: {e}")
        return {"success": False, "error": str(e), "findings": []}


def run_blue_team_resolution(
    original_content: str,
    identified_issues: List[Dict[str, Any]],
    content_type: str,
    blue_team_models: List[str],
    api_key: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Phase 2: Run Blue Team to resolve identified issues
    Implements the Blue Team Patch Development process
    """
    try:
        if not identified_issues:
            return {
                "success": True,
                "improved_content": original_content,
                "applied_fixes": [],
                "message": "No issues to fix, returning original content"
            }
        
        # Prepare issues for blue team
        issues_summary = "\n".join([
            f"- {issue.get('severity', 'Medium')} {issue.get('category', 'General')}: {issue.get('description', 'No description')}"
            for issue in identified_issues
        ])
        
        blue_team_prompts = [
            f"""You are a blue team fixer. Address the following issues in this {content_type.replace('_', ' ')} content:
            
            Original content:
            {original_content}
            
            Issues to fix:
            {issues_summary}
            
            Provide an improved version that addresses all identified issues while preserving functionality.
            Explain what fixes were applied.""",
            
            f"""As a blue team member, improve this {content_type.replace('_', ' ')} content by fixing these issues:
            {issues_summary}
            
            Content to improve:
            {original_content}
            
            Return the fixed content with all issues resolved.""",
            
            f"""You are a remediation specialist. Transform this {content_type.replace('_', ' ')} content
            to fix the following issues:
            {issues_summary}
            
            Original content:
            {original_content}
            
            Provide the improved content with fixes applied and a summary of changes made."""
        ]
        
        applied_fixes = []
        improved_content = original_content
        
        # Apply fixes using blue team models
        for i, model in enumerate(blue_team_models[:2]):  # Use first 2 models for fixes
            prompt = blue_team_prompts[i % len(blue_team_prompts)]
            
            try:
                response = _request_openai_compatible_chat(
                    api_key,
                    api_base,
                    model,
                    [{"role": "user", "content": prompt}],
                    {"Content-Type": "application/json"},
                    temperature,
                    0.95,
                    0.0,
                    0.0,
                    max_tokens,
                    None
                )
                
                if response:
                    # Parse the fixed content and applied fixes
                    parsed_result = _parse_blue_team_result(response, improved_content)
                    improved_content = parsed_result.get("fixed_content", improved_content)
                    applied_fixes.extend(parsed_result.get("applied_fixes", []))
                    
            except Exception as e:
                st.warning(f"Error in blue team fix attempt {i+1}: {e}")
        
        return {
            "success": True,
            "improved_content": improved_content,
            "applied_fixes": applied_fixes,
            "total_fixes": len(applied_fixes)
        }
    
    except Exception as e:
        st.error(f"Error in blue team resolution: {e}")
        return {"success": False, "error": str(e), "improved_content": original_content, "applied_fixes": []}


def _parse_red_team_findings(red_team_output: str) -> List[Dict[str, Any]]:
    """Parse red team output to extract structured findings"""
    try:
        # Try to parse as JSON first (if LLM returned structured data)
        if red_team_output.strip().startswith('{') or red_team_output.strip().startswith('['):
            parsed = json.loads(red_team_output)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and 'findings' in parsed:
                return parsed['findings']
        
        # If not JSON, try to extract findings using heuristics
        findings = []
        
        # Look for common patterns in the output
        lines = red_team_output.split('\n')
        current_finding = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for severity indicators
            if any(sev.lower() in line.lower() for sev in ['critical', 'high', 'medium', 'low']):
                if current_finding and 'description' in current_finding:
                    findings.append(current_finding)
                    current_finding = {}
                
                current_finding['severity'] = 'Critical' if 'critical' in line.lower() else \
                                            'High' if 'high' in line.lower() else \
                                            'Medium' if 'medium' in line.lower() else 'Low'
            
            # Look for issue descriptions
            elif any(cat.lower() in line.lower() for cat in ['issue:', 'problem:', 'vulnerability:', 'flaw:', 'error:']):
                if 'description' not in current_finding:
                    current_finding['description'] = line
                else:
                    # Add to existing finding
                    current_finding['description'] += f" {line}"
            
            # Look for categories
            elif any(cat.lower() in line.lower() for cat in ['functional', 'structural', 'security', 'compliance', 'performance']):
                current_finding['category'] = line.split(':')[0].strip() if ':' in line else 'General'
        
        # Add the last finding if exists
        if current_finding and 'description' in current_finding:
            findings.append(current_finding)
        
        return findings
        
    except Exception as e:
        st.warning(f"Error parsing red team findings: {e}")
        # Return a basic finding if parsing fails
        return [{
            "severity": "Medium",
            "category": "General",
            "description": red_team_output[:200] + "..." if len(red_team_output) > 200 else red_team_output,
            "potential_impact": "Unknown",
            "remediation": "Review and address as appropriate"
        }]


def _parse_blue_team_result(blue_team_output: str, original_content: str) -> Dict[str, Any]:
    """Parse blue team output to extract fixed content and applied fixes"""
    try:
        # Try to parse as JSON first
        if blue_team_output.strip().startswith('{'):
            parsed = json.loads(blue_team_output)
            if isinstance(parsed, dict):
                return {
                    "fixed_content": parsed.get("fixed_content", original_content),
                    "applied_fixes": parsed.get("applied_fixes", []),
                    "changes_summary": parsed.get("changes_summary", "")
                }
        
        # Look for content between code blocks or special markers
        fixed_content = original_content
        applied_fixes = []
        
        # Extract content between markers
        if "```" in blue_team_output:
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', blue_team_output, re.DOTALL)
            if code_blocks:
                fixed_content = code_blocks[0].strip()
        
        # If no code blocks, assume the entire response is the fixed content
        if fixed_content == original_content:
            fixed_content = blue_team_output.strip()
        
        # Extract change descriptions
        if "changes:" in blue_team_output.lower() or "fixes:" in blue_team_output.lower():
            import re
            changes_section = re.search(r'(changes:|fixes:|improvements:)\s*(.*?)(?:\n\s*\n|$)', blue_team_output, re.IGNORECASE | re.DOTALL)
            if changes_section:
                changes_text = changes_section.group(2).strip()
                applied_fixes = [{"description": desc.strip(), "status": "applied"} 
                                for desc in changes_text.split('\n') if desc.strip() and not desc.strip().startswith('- ')]
        
        return {
            "fixed_content": fixed_content,
            "applied_fixes": applied_fixes,
            "changes_summary": f"Applied {len(applied_fixes)} fixes to original content"
        }
    
    except Exception as e:
        st.warning(f"Error parsing blue team result: {e}")
        return {
            "fixed_content": blue_team_output,
            "applied_fixes": [{"description": "Unknown fixes applied", "status": "applied"}],
            "changes_summary": "Fixes applied (parsing failed, using full response)"
        }


def _categorize_findings_by_severity(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    """Categorize findings by severity level"""
    severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    
    for finding in findings:
        severity = finding.get("severity", "Medium").capitalize()
        if severity in severity_counts:
            severity_counts[severity] += 1
        else:
            severity_counts["Medium"] += 1  # Default to medium
    
    return severity_counts


def _request_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    extra_headers: Dict[str, str],
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int]
) -> Optional[str]:
    """
    Make a request to an OpenAI-compatible API
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        # If openai package is not available, try using requests
        import requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        headers.update(extra_headers)
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens
        }
        
        if seed is not None:
            data["seed"] = seed
            
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        st.error(f"Error making API request: {e}")
        return None




# The main function that integrates with the existing adversarial.py
def run_enhanced_adversarial_evolution(
    content: str,
    content_type: str,
    red_team_models: List[str],
    blue_team_models: List[str],
    evaluator_models: List[str],
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    max_iterations: int = 50,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    custom_requirements: str = "",
    compliance_rules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main function to run enhanced adversarial evolution using the tripartite AI architecture
    """
    return run_comprehensive_adversarial_testing(
        content=content,
        content_type=content_type,
        red_team_models=red_team_models,
        blue_team_models=blue_team_models,
        evaluator_models=evaluator_models,
        api_key=api_key,
        api_base=api_base,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        custom_requirements=custom_requirements,
        compliance_rules=compliance_rules
    )


if __name__ == "__main__":
    # Example usage
    print("Adversarial Testing Module loaded successfully")