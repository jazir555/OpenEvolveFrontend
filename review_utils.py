def determine_review_type(content: str) -> str:
    """
    Determine the review type based on content analysis.
    
    Args:
        content: The content to analyze
        
    Returns:
        str: The determined review type (code, document, plan, etc.)
    """
    content_lower = content.lower()
    
    # Check for code indicators
    code_indicators = ['def ', 'function', 'class ', 'import ', 'from ', 'public ', 'private ', 'var ', 'let ', 'const ']
    document_indicators = ['section', 'clause', 'paragraph', 'article', 'chapter']
    plan_indicators = ['schedule', 'timeline', 'resource', 'budget', 'milestone', 'objective']
    
    if any(indicator in content_lower for indicator in code_indicators):
        return "code"
    elif any(indicator in content_lower for indicator in document_indicators):
        return "document"
    elif any(indicator in content_lower for indicator in plan_indicators):
        return "plan"
    else:
        return "general"


def get_appropriate_prompts(review_type: str) -> tuple:
    """
    Get appropriate red team and blue team prompts based on review type.
    
    Args:
        review_type: The type of review (code, document, plan, etc.)
        
    Returns:
        tuple: (red_team_prompt, blue_team_prompt)
    """
    if review_type == "code":
        red_prompt = "Analyze this code for security vulnerabilities, performance issues, and logical errors."
        blue_prompt = "Fix identified issues in the code, optimize performance, and improve code quality."
    elif review_type == "document":
        red_prompt = "Review this document for accuracy, clarity, and compliance with standards."
        blue_prompt = "Improve the document's clarity, accuracy, and ensure compliance with standards."
    elif review_type == "plan":
        red_prompt = "Critique this plan for feasibility, resource allocation, and risk management."
        blue_prompt = "Enhance the plan to address risks, optimize resources, and improve feasibility."
    else:
        red_prompt = "Critique this content for weaknesses, inconsistencies, and areas of improvement."
        blue_prompt = "Improve the content addressing the identified weaknesses and inconsistencies."
    
    return red_prompt, blue_prompt
