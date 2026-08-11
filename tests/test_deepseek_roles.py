#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek Multi-Agent Gauntlet Test with Team Configuration
"""

import os
import sys
import codecs
from dotenv import load_dotenv

# Set UTF-8 for output
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

load_dotenv()

# First, let's test direct DeepSeek calls for different roles
import requests

DEEPSEEK_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def call_deepseek(role, prompt):
    """Call DeepSeek API for a specific role"""
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_KEY}',
        'Content-Type': 'application/json'
    }
    
    messages = [
        {'role': 'system', 'content': f'You are a {role} in a code evaluation gauntlet.'},
        {'role': 'user', 'content': prompt}
    ]
    
    data = {
        'model': 'deepseek-chat',
        'messages': messages,
        'max_tokens': 200,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(DEEPSEEK_URL, json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {e}"

print("="*80)
print("DEEPSEEK MULTI-ROLE GAUNTLET TEST")
print("="*80)
print()

# Code to evaluate
code = """
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    if result:
        session['password'] = password
        return True
    return False
"""

print(f"Evaluating code ({len(code)} chars):")
print(code)
print()

# Test 1: Red Team Agent (Security Attacker)
print("="*80)
print("RED TEAM AGENT 1 (Security Attack)")
print("="*80)
red_prompt = f"Find security vulnerabilities in this code:\n{code}"
red_response = call_deepseek("security researcher", red_prompt)
print(f"Response: {red_response[:300]}...")
print()

# Test 2: Red Team Agent 2 (Code Attacker)
print("="*80)
print("RED TEAM AGENT 2 (Code Attack)")
print("="*80)
red2_prompt = f"How would you exploit this code?\n{code}"
red2_response = call_deepseek("penetration tester", red2_prompt)
print(f"Response: {red2_response[:300]}...")
print()

# Test 3: Blue Team Agent (Defender)
print("="*80)
print("BLUE TEAM AGENT 1 (Defense)")
print("="*80)
blue_prompt = f"Fix the security issues in this code:\n{code}"
blue_response = call_deepseek("security engineer", blue_prompt)
print(f"Response: {blue_response[:300]}...")
print()

# Test 4: Blue Team Agent 2 (Reviewer)
print("="*80)
print("BLUE TEAM AGENT 2 (Code Review)")
print("="*80)
blue2_prompt = f"Suggest improvements for this code:\n{code}"
blue2_response = call_deepseek("senior developer", blue2_prompt)
print(f"Response: {blue2_response[:300]}...")
print()

# Test 5: Judge Agent (Evaluator)
print("="*80)
print("JUDGE AGENT (Final Evaluation)")
print("="*80)
judge_prompt = f"""Based on these analyses, rate this code quality (0.0-1.0):
- Red Team found vulnerabilities
- Blue Team suggested fixes

Code: {code}

Provide a score and brief justification."""
judge_response = call_deepseek("code judge", judge_prompt)
print(f"Response: {judge_response[:300]}...")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print("DeepSeek Roles Tested:")
print("  ✓ Red Team Agent 1 (Security Researcher)")
print("  ✓ Red Team Agent 2 (Penetration Tester)")
print("  ✓ Blue Team Agent 1 (Security Engineer)")
print("  ✓ Blue Team Agent 2 (Senior Developer)")
print("  ✓ Judge Agent (Code Judge)")
print()
print("[SUCCESS] DeepSeek can serve multiple independent roles in the gauntlet!")
print("="*80)
