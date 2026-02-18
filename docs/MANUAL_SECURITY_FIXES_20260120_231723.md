# 🔧 Manual Security Fix Report
# Generated: 2026-01-20 23:17:23

## Issues Requiring Manual Fix

### 1. Pickle Usage (1657 issues)

**Action Required:** Replace `pickle` with `json` for secure serialization

**File:** ace_knowledge_artifacts.py
**Line:** 129
**Code:** `- __getstate__ and __setstate__ for pickle support (locks are not serializable)`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** ace_knowledge_artifacts.py
**Line:** 154
**Code:** `# SERIALIZATION FIX: Add pickle support for locks`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** ace_knowledge_artifacts.py
**Line:** 163
**Code:** `"""Restore state from pickle, recreating the lock."""`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** ace_knowledge_artifacts.py
**Line:** 845
**Code:** `# SERIALIZATION FIX: Add pickle support for locks`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** ace_knowledge_artifacts.py
**Line:** 854
**Code:** `"""Restore state from pickle, recreating the lock."""`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** advanced_cache.py
**Line:** 18
**Code:** `import pickle`
**Fix:** Replace with import json

**File:** advanced_cache.py
**Line:** 127
**Code:** `size = sys.getsizeof(pickle.dumps(value))`
**Fix:** Replace with json.dump() and change file format from .pkl to .json

**File:** advanced_cache.py
**Line:** 311
**Code:** `value = pickle.loads(value_blob)`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** advanced_cache.py
**Line:** 339
**Code:** `value_blob = pickle.dumps(value)`
**Fix:** Replace with json.dump() and change file format from .pkl to .json

**File:** advanced_unit_tests_comprehensive.py
**Line:** 32
**Code:** `import pickle`
**Fix:** Replace with import json

**File:** auto_fix_security.py
**Line:** 8
**Code:** `- Insecure pickle usage (B301)`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 138
**Code:** `"""Fix hardcoded temp directories and insecure pickle usage."""`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 159
**Code:** `# Check for pickle.load (B301)`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** auto_fix_security.py
**Line:** 163
**Code:** `node.func.value.id == 'pickle'):`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 165
**Code:** `logger.critical(f"  [{self.filename}] pickle.load() at line {node.lineno} - MANUAL FIX REQUIRED")`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** auto_fix_security.py
**Line:** 167
**Code:** `'type': 'insecure_pickle',`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 169
**Code:** `'fix': 'CRITICAL: Replace pickle.load() with json.load() - MANUAL FIX REQUIRED'`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** auto_fix_security.py
**Line:** 306
**Code:** `'pickle_usage': 0,`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 325
**Code:** `# Check for pickle import or usage`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 326
**Code:** `if 'pickle' in line and ('import' in line or 'pickle.' in line):`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 327
**Code:** `issues['pickle_usage'] += 1`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 377
**Code:** `'pickle_usage': 0,`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 394
**Code:** `logger.info(f"  Pickle usage: {total_issues['pickle_usage']}")`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 442
**Code:** `manual_fixes = total_issues['pickle_usage'] + total_issues['hardcoded_tmp'] + total_issues['certificate_issues']`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** auto_fix_security.py
**Line:** 445
**Code:** `logger.info(f"  - Replace pickle.load() with json.load(): {total_issues['pickle_usage']}")`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** blue_team_coordinator.py
**Line:** 31
**Code:** `import pickle`
**Fix:** Replace with import json

**File:** blue_team_coordinator.py
**Line:** 966
**Code:** `pickle.dump(state, f)`
**Fix:** Replace with json.dump() and change file format from .pkl to .json

**File:** blue_team_coordinator.py
**Line:** 979
**Code:** `state = pickle.load(f)`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** check_broken_imports.py
**Line:** 102
**Code:** `'string', 'secrets', 'gc', 'tracemalloc', 'weakref', 'pickle',`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** check_root_imports.py
**Line:** 70
**Code:** `'string', 'secrets', 'gc', 'tracemalloc', 'weakref', 'pickle',`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** evaluator_team_coordinator.py
**Line:** 37
**Code:** `import pickle`
**Fix:** Replace with import json

**File:** evaluator_team_coordinator.py
**Line:** 1662
**Code:** `pickle.dump(state, f)`
**Fix:** Replace with json.dump() and change file format from .pkl to .json

**File:** evaluator_team_coordinator.py
**Line:** 1673
**Code:** `state = pickle.load(f)`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** fix_manual_security_issues.py
**Line:** 35
**Code:** `def find_pickle_usage(self) -> List[Dict]:`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 36
**Code:** `"""Find all pickle usage that needs to be replaced with JSON."""`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 52
**Code:** `if 'pickle' in line:`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 57
**Code:** `'issue': 'pickle_usage',`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 58
**Code:** `'recommendation': self._suggest_pickle_fix(line)`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 65
**Code:** `def _suggest_pickle_fix(self, line: str) -> str:`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 66
**Code:** `"""Suggest fix for pickle usage."""`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 67
**Code:** `if 'pickle.load' in line:`
**Fix:** Replace with json.load() and change file format from .pkl to .json

**File:** fix_manual_security_issues.py
**Line:** 69
**Code:** `elif 'pickle.dump' in line:`
**Fix:** Replace with json.dump() and change file format from .pkl to .json

**File:** fix_manual_security_issues.py
**Line:** 71
**Code:** `elif 'import pickle' in line:`
**Fix:** Replace with import json

**File:** fix_manual_security_issues.py
**Line:** 74
**Code:** `return "Review pickle usage and consider replacing with JSON"`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 150
**Code:** `pickle_issues = self.find_pickle_usage()`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 162
**Code:** `if pickle_issues:`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 163
**Code:** `report_lines.append(f"### 1. Pickle Usage ({len(pickle_issues)} issues)")`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 165
**Code:** `report_lines.append("**Action Required:** Replace `pickle` with `json` for secure serialization")`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 167
**Code:** `for issue in pickle_issues[:50]:  # Limit to first 50`
**Fix:** Review pickle usage and consider replacing with JSON

**File:** fix_manual_security_issues.py
**Line:** 174
**Code:** `if len(pickle_issues) > 50:`
**Fix:** Review pickle usage and consider replacing with JSON

*... and 1607 more pickle issues*

### 2. Hardcoded Temp Paths (195 issues)

**Action Required:** Replace hardcoded `/tmp/` with `tempfile` module

**File:** add_class_function_docstrings.py
**Line:** 220
**Code:** `>>> store = FileCheckpointStore(base_path="/tmp/checkpoints")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** auto_fix_security.py
**Line:** 177
**Code:** `if isinstance(node.args[0].value, str) and '/tmp/' in node.args[0].value:`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** auto_fix_security.py
**Line:** 330
**Code:** `if "'/tmp/" in line or '"/tmp/' in line:`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** deployment_operations.py
**Line:** 285
**Code:** `tar.extractall(path='/tmp/sovereign_restore')`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** deployment_operations.py
**Line:** 288
**Code:** `backup_db = '/tmp/sovereign_restore/database.db'`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** deployment_operations.py
**Line:** 294
**Code:** `backup_config = '/tmp/sovereign_restore/config'`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** fix_manual_security_issues.py
**Line:** 109
**Code:** `if 'open(' in line and '/tmp/' in line:`
**Fix:** Use tempfile.NamedTemporaryFile() or tempfile.mkstemp() instead

**File:** fix_manual_security_issues.py
**Line:** 111
**Code:** `elif '/tmp/' in line and '=' in line:`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** fix_manual_security_issues.py
**Line:** 241
**Code:** `report_lines.append("temp_dir = '/tmp/myapp_data'")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** maker_engine.py
**Line:** 362
**Code:** `>>> store = FileCheckpointStore(path="/tmp/checkpoint.json")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** agentic-context-engine\benchmarks\base.py
**Line:** 131
**Code:** `data_dir = os.getenv("BENCHMARK_DATA_DIR", "/tmp/benchmark_data")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 10
**Code:** `test_file = "/tmp/test_file.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 26
**Code:** `test_file = "/tmp/test_file.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 50
**Code:** `result = tool._run(file_path="/tmp/no_permission.txt")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 56
**Code:** `test_file1 = "/tmp/test1.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 57
**Code:** `test_file2 = "/tmp/test2.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 75
**Code:** `test_file = "/tmp/multiline_test.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 116
**Code:** `test_file = "/tmp/short_test.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 135
**Code:** `test_file = "/tmp/negative_test.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\crewAI-main\lib\crewai-tools\tests\rag\test_docx_loader.py
**Line:** 55
**Code:** `mock_temp = Mock(name="/tmp/temp_docx_file.docx")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 10
**Code:** `test_file = "/tmp/test_file.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 26
**Code:** `test_file = "/tmp/test_file.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 50
**Code:** `result = tool._run(file_path="/tmp/no_permission.txt")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 56
**Code:** `test_file1 = "/tmp/test1.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 57
**Code:** `test_file2 = "/tmp/test2.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 75
**Code:** `test_file = "/tmp/multiline_test.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 116
**Code:** `test_file = "/tmp/short_test.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\file_read_tool_test.py
**Line:** 135
**Code:** `test_file = "/tmp/negative_test.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** crewAI\lib\crewai-tools\tests\rag\test_docx_loader.py
**Line:** 55
**Code:** `mock_temp = Mock(name="/tmp/temp_docx_file.docx")`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** Curie\benchmark\exp_bench\evaluation\eval.py
**Line:** 389
**Code:** `elif path.startswith("/tmp/"): # an edge case we noticed when the setup extractor agent did not know where the repo is, and cloned their own repo in /tmp of the existing filesystem. We assume that not github repo will have a tmp folder..`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** Curie\benchmark\exp_bench\evaluation\eval.py
**Line:** 390
**Code:** `# Remove "/tmp/" prefix`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** Curie\benchmark\exp_bench\evaluation\eval.py
**Line:** 391
**Code:** `return path[len("/tmp/"):]`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** Curie\benchmark\exp_bench\evaluation\judge.py
**Line:** 507
**Code:** `elif path.startswith("/tmp/"): # an edge case we noticed when the setup extractor agent did not know where the repo is, and cloned their own repo in /tmp of the existing filesystem. We assume that not github repo will have a tmp folder..`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** Curie\benchmark\exp_bench\evaluation\judge.py
**Line:** 508
**Code:** `# Remove "/tmp/" prefix`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** Curie\benchmark\exp_bench\evaluation\judge.py
**Line:** 509
**Code:** `return path[len("/tmp/"):]`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** CrewAI\scripts\bootstrap_project.py
**Line:** 15
**Code:** `--worktrees "/tmp/crewai_worktrees" \`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** CrewAI\scripts\create_test_tickets.py
**Line:** 31
**Code:** `phases_folder_path="/tmp/phases",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\scripts\create_test_tickets_sql.py
**Line:** 24
**Code:** `(workflow_id, "E2E Test Workflow", "/tmp/phases", "active", datetime.utcnow().isoformat()),`
**Fix:** Replace hardcoded /tmp with tempfile module functions

**File:** CrewAI\src\agents\manager.py
**Line:** 272
**Code:** `debug_prompt_path = f"/tmp/crewai_debug_prompt_{agent_id}.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\src\core\simple_config.py
**Line:** 47
**Code:** `self.worktree_base_path = Path(paths.get('worktree_base', '/tmp/crewai_worktrees'))`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\src\interfaces\cli_interface.py
**Line:** 115
**Code:** `prompt_file = f"/tmp/hep_prompt_{task_id}.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\src\interfaces\cli_interface.py
**Line:** 208
**Code:** `prompt_file = f"/tmp/opencode_prompt_{task_id}.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\src\interfaces\cli_interface.py
**Line:** 423
**Code:** `prompt_file = f"/tmp/hep_prompt_{kwargs.get('task_id', 'default')}.txt"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\src\sdk\config.py
**Line:** 47
**Code:** `worktree_base: str = "/tmp/crewai_worktrees"`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\tests\conftest.py
**Line:** 245
**Code:** `working_directory="/tmp/test-project",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\tests\test_agent_workflow_context.py
**Line:** 479
**Code:** `phases_folder_path="/tmp/test",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\tests\test_diagnostic_agent.py
**Line:** 102
**Code:** `phases_folder_path="/tmp/test",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\tests\test_diagnostic_agent.py
**Line:** 202
**Code:** `result_file_path="/tmp/result.md",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\tests\test_diagnostic_integration.py
**Line:** 134
**Code:** `phases_folder_path="/tmp/test",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

**File:** CrewAI\tests\test_multi_workflow.py
**Line:** 83
**Code:** `phases_folder_path="/tmp/test",`
**Fix:** Replace with tempfile.mkdtemp(prefix='yourprefix_')

*... and 145 more temp path issues*

### 3. Certificate Verification Issues (41 issues)

**Action Required:** Remove `verify=False` to enable SSL certificate validation

**File:** auto_fix_security.py
**Line:** 334
**Code:** `if 'verify=False' in line:`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** fix_manual_security_issues.py
**Line:** 133
**Code:** `if 'verify=False' in line or 'verify = False' in line:`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** fix_manual_security_issues.py
**Line:** 139
**Code:** `'recommendation': "Remove verify=False or set verify=True for SSL certificate validation"`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** fix_manual_security_issues.py
**Line:** 199
**Code:** `report_lines.append("**Action Required:** Remove `verify=False` to enable SSL certificate validation")`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** fix_manual_security_issues.py
**Line:** 253
**Code:** `report_lines.append("response = requests.get(url, verify=False)  # Disables SSL validation")`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** agentic-context-engine\tests\test_instructor_integration.py
**Line:** 671
**Code:** `base_llm = LiteLLMClient(model="gpt-4", ssl_verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** agentic-context-engine\tests\test_litellm_client.py
**Line:** 301
**Code:** `"""Test ssl_verify=False is passed through."""`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** agentic-context-engine\tests\test_litellm_client.py
**Line:** 306
**Code:** `agent = ACELiteLLM(model="gpt-4", ssl_verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** agentic-context-engine\tests\test_litellm_client.py
**Line:** 374
**Code:** `"""Test ssl_verify=False is included in LiteLLM call_params."""`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** agentic-context-engine\tests\test_litellm_client.py
**Line:** 379
**Code:** `client = LiteLLMClient(model="gpt-4", ssl_verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\numpy\lib\_ufunclike_impl.py
**Line:** 16
**Code:** `@array_function_dispatch(_dispatcher, verify=False, module='numpy')`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\numpy\lib\_ufunclike_impl.py
**Line:** 70
**Code:** `@array_function_dispatch(_dispatcher, verify=False, module='numpy')`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\numpy\lib\_ufunclike_impl.py
**Line:** 140
**Code:** `@array_function_dispatch(_dispatcher, verify=False, module='numpy')`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\numpy\_core\multiarray.py
**Line:** 104
**Code:** `module='numpy', docs_from_dispatcher=True, verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\numpy\_core\tests\test_overrides.py
**Line:** 300
**Code:** `@array_function_dispatch(lambda x: (x,), verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pandas\core\algorithms.py
**Line:** 807
**Code:** `verify=False,`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pandas\core\algorithms.py
**Line:** 1481
**Code:** `codes equal to ``-1``. If ``verify=False``, it is assumed there`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pandas\core\frame.py
**Line:** 7227
**Code:** `indexer, axis=self._get_block_manager_axis(axis), verify=False`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pandas\core\frame.py
**Line:** 12224
**Code:** `res = data._mgr.take(indexer[q_idx], verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pandas\core\generic.py
**Line:** 5346
**Code:** `new_data = self._mgr.take(indexer, axis=baxis, verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pandas\core\internals\managers.py
**Line:** 882
**Code:** `Pass verify=False if this check has been done by the caller.`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pip\_internal\network\session.py
**Line:** 304
**Code:** `super().cert_verify(conn=conn, url=url, verify=False, cert=cert)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pip\_internal\network\session.py
**Line:** 315
**Code:** `super().cert_verify(conn=conn, url=url, verify=False, cert=cert)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pip\_vendor\distlib\wheel.py
**Line:** 175
**Code:** `def __init__(self, filename=None, sign=False, verify=False):`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** openevolve_test_env\Lib\site-packages\pip\_vendor\urllib3\contrib\securetransport.py
**Line:** 795
**Code:** `self._verify = False`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\examples\demo.py
**Line:** 71
**Code:** `verify=False,  # Skip Lean 4 verification for demo`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\examples\demo.py
**Line:** 144
**Code:** `config = PSI3Config(mode="fast", verify=False, verbose=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\examples\demo.py
**Line:** 215
**Code:** `config = PSI3Config(mode="fast", verify=False, verbose=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\tests\unit\test_constraint_inverter.py
**Line:** 452
**Code:** `config = PSI3Config(mode="fast", verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\tests\unit\test_constraint_inverter.py
**Line:** 471
**Code:** `config = PSI3Config(mode="fast", verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\tests\unit\test_constraint_inverter.py
**Line:** 513
**Code:** `config = PSI3Config(mode="fast", verify=False, verbose=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\tests\unit\test_constraint_inverter.py
**Line:** 537
**Code:** `config = PSI3Config(mode="fast", verify=False, verbose=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** phase2\psi3\tests\unit\test_constraint_inverter.py
**Line:** 565
**Code:** `config = PSI3Config(mode="fast", verify=False, verbose=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\demos\demos_databases_apis\alienvault\unimatrix.py
**Line:** 77
**Code:** `#verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\demos\demos_databases_apis\alienvault\unimatrix.py
**Line:** 101
**Code:** `#response = self.client.get(self.url+item_type, params=params, verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\graphistry\tests\test_certificate_validation_session.py
**Line:** 55
**Code:** `"""Verify chain_remote passes verify=False when certificate_validation=False"""`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\graphistry\tests\test_certificate_validation_session.py
**Line:** 85
**Code:** `# Verify that verify=False was passed to requests.post`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\graphistry\tests\test_certificate_validation_session.py
**Line:** 125
**Code:** `"""Verify python_remote passes verify=False when certificate_validation=False"""`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\graphistry\tests\test_certificate_validation_session.py
**Line:** 155
**Code:** `# Verify that verify=False was passed to requests.post`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\graphistry\tests\test_certificate_validation_session.py
**Line:** 207
**Code:** `# Call chain_remote for client2 (should use verify=False)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation

**File:** pygraphistry\graphistry\tests\test_certificate_validation_session.py
**Line:** 247
**Code:** `# Call with client (should use client's verify=False, not global's True)`
**Fix:** Remove verify=False or set verify=True for SSL certificate validation


## Summary

- Total pickle usage issues: 1657
- Total hardcoded temp path issues: 195
- Total certificate verification issues: 41
- **Total manual fixes required: 1893**

## Fix Examples

### Pickle → JSON
```python
# ❌ BEFORE (insecure)
import pickle
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)  # Can execute arbitrary code!

# ✅ AFTER (secure)
import json
with open('data.json', 'r') as f:
    data = json.load(f)  # Safe, no code execution
```

### Hardcoded /tmp → tempfile
```python
# ❌ BEFORE (insecure)
temp_dir = '/tmp/myapp_data'
os.makedirs(temp_dir, exist_ok=True)

# ✅ AFTER (secure)
import tempfile
temp_dir = tempfile.mkdtemp(prefix='myapp_')
```

### Certificate Verification
```python
# ❌ BEFORE (insecure)
response = requests.get(url, verify=False)  # Disables SSL validation

# ✅ AFTER (secure)
response = requests.get(url, verify=True)  # Enable SSL validation
# Or for custom CA bundle:
response = requests.get(url, verify='/path/to/ca-bundle.crt')
```
