from typing import Dict, Any

CONFIG_PROFILES: Dict[str, Dict[str, Any]] = {
    "Security Hardening": {
        "system_prompt": "You are a security expert focused on making protocols robust against attacks. Focus on identifying and closing security gaps, enforcing least privilege, and adding comprehensive error handling.",
        "evaluator_system_prompt": "You are a security auditor evaluating the protocol for vulnerabilities and weaknesses.",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_iterations": 15,
        "adversarial_confidence": 95,
        "adversarial_min_iter": 5,
        "adversarial_max_iter": 20,
    },
    "Compliance Focus": {
        "system_prompt": "You are a compliance expert ensuring protocols meet regulatory requirements. Focus on completeness, auditability, and regulatory alignment.",
        "evaluator_system_prompt": "You are a compliance auditor checking if the protocol meets all necessary regulatory requirements.",
        "temperature": 0.5,
        "top_p": 0.8,
        "max_iterations": 10,
        "adversarial_confidence": 90,
        "adversarial_min_iter": 3,
        "adversarial_max_iter": 15,
    },
    "Operational Efficiency": {
        "system_prompt": "You are an operations expert focused on making protocols efficient and practical. Focus on streamlining processes while maintaining effectiveness.",
        "evaluator_system_prompt": "You are an operations expert evaluating the protocol for practicality and efficiency.",
        "temperature": 0.6,
        "top_p": 0.85,
        "max_iterations": 12,
        "adversarial_confidence": 85,
        "adversarial_min_iter": 3,
        "adversarial_max_iter": 12,
    },
    "Beginner-Friendly": {
        "system_prompt": "You are helping a beginner write clear, understandable protocols. Focus on clarity, simplicity, and completeness.",
        "evaluator_system_prompt": "You are evaluating if the protocol is clear and understandable for beginners.",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_iterations": 8,
        "adversarial_confidence": 80,
        "adversarial_min_iter": 2,
        "adversarial_max_iter": 10,
    },
}
