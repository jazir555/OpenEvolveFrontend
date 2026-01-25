import re
import traceback
import time
import numpy as np
from client import Lean4Client, batch_verify_proof

def get_formal_statement_prompt(informal_problem: str, header: str = "import Mathlib\n") -> str:
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt += informal_problem
    prompt += f"\n\nYour code should start with:\n```Lean4\n{header}\n```\n"
    return prompt

def extract_after_last_think(text):
    tag = "</think>"
    idx = text.rfind(tag)
    if idx != -1:
        return text[idx + len(tag):].strip()
    else:
        return text

def extract_the_last_lean_code(output):
	if "```" not in output:
		return output
	result = []
	ok = False
	for line in list(output.splitlines())[::-1]:
		if "```" in line:
			if ok: break
			ok = True
			continue
		if ok: result.append(line)
	result = '\n'.join(result[::-1])
	return result

def extract_main_theorem(output):
	lines = output.splitlines()
	for i, line in enumerate(lines):
		if line.startswith("theorem"):
			break
	else:
		return output
	return '\n'.join(lines[i:])

def post_process_formal_statement_with_reasoning(output, header):
	output = extract_after_last_think(output)
	output = extract_the_last_lean_code(output)
	if output.startswith(header):
		return output
	return header + "\n\n" + extract_main_theorem(output)

def remove_before_theorem(s: str) -> str:
	lines = s.splitlines()
	for i, line in enumerate(lines):
		line = line.lstrip()
		if line.startswith("theorem") or line.startswith("def"):
			return "\n".join(lines[i:])
	return "<No theorem>"

def get_p2q_proof_by_exact(p: str, q: str, prefix: str) -> str:
    original_q = q
    if not prefix:
        q = remove_before_theorem(q)
    else:
        p = p.removeprefix(prefix)
        q = q.removeprefix(prefix)
    q = re.sub(r':=(\s*(by)*\n*)*sorry', ':= by', q).strip()
    if not q.endswith(':= by'):
        return f"Invalid proof: {original_q}"
    if not prefix:
        code = p + '\n\n' + q + '\n' + 'exact?' + '\n'
    else:
        code = prefix + "\n\n" + p + '\n\n' + q + '\n' + 'exact?' + '\n'
    return code

THEOREM_NAME_PATTERN = re.compile('theorem\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*')

def extract_theorem_name(lean_code: str) -> str:
    match = THEOREM_NAME_PATTERN.search(lean_code)
    if match:
        return match.group(1)
    assert False, f"No theorem name in:\n {lean_code}"

def replace_theorem_name(lean_code: str, new_name: str) -> str:
    return THEOREM_NAME_PATTERN.sub(rf'theorem {new_name} ', lean_code, count=1)

def judge_beq_correctness(result: dict[str, str], p: str) -> bool:
    if not result["pass"]:
        return False
    try:
        name = extract_theorem_name(p)
        last_info = str([m for m in result["infos"] if m["severity"] == "info"][-1]["data"])
        if 'Try this: exact' in last_info and name in last_info:
            return True
    except Exception as err:
        pass
    return False

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if c == 0:
        if k > n:
            print(f"Warning: filling {k - n} entries with unsuccessful")
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

class KiminaLeanClient:
    BACKOFF = 3
    def __init__(self, host: str, port: int, timeout: int, max_workers: int):
        self._url         = f"http://{host}:{port}"
        self._timeout     = timeout
        self._max_workers = max_workers
        self._client      = Lean4Client(self._url, disable_cache=False)

    @property
    def url(self) -> str:
        return self._url

    def collect_results(self, code: str, result: dict[str, str]) -> dict[str, str]:
        if "response" not in result or not isinstance(result["response"], dict):
            return {
                "pass": False,
                "complete": False,
                "system_errors": result.get("error", traceback.format_exc()),
                "time": result.get("time", None)
            }
        response = result["response"]
        result = {
            "sorries"       : response.get('sorries', []), 
            "tactics"       : response.get('tactics', []),
            "errors"        : [m for m in response.get('messages', []) if m['severity'] == 'error'],
            "warnings"      : [m for m in response.get('messages', []) if m['severity'] == 'warning'],
            "infos"         : [m for m in response.get('messages', []) if m['severity'] == 'info'],
            "system_errors" : result.get("error", None),
            "verified_code" : code,
            "time"          : response.get('time', None) 
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
        return result
    
    def _query_impl(self, code_list: list[str], is_clear_cache: bool = False, disable_process_bar: bool = False) -> dict[str, str]:
        samples  = [{"proof": code, "custom_id": str(i)} for i, code in enumerate(code_list)]
        if is_clear_cache:
            self._client.clear_cache()
        response = batch_verify_proof(
            client              = self._client,
            samples             = samples,
            timeout             = self._timeout,
            num_proc            = self._max_workers,
            batch_size          = 1,
            disable_process_bar = disable_process_bar
        )
        results = sorted(response, key=lambda x : int(x["custom_id"]))
        return [self.collect_results(code, result) for code, result in zip(code_list, results)]

    def query(self, **kwargs):
        curr_backoff = 3
        while True:
            try:
                return self._query_impl(**kwargs)
            except Exception as err:
                print(f"Error in {self.__class__.__name__}: {err}, waiting {curr_backoff} seconds and then retrying...")
                time.sleep(curr_backoff)
                continue

