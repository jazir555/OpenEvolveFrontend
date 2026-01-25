from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import eval_utils as utils 

MODEL_DIR = "/path/to/Stepfun-Formalizer/"

if __name__ == "__main__":

    system_prompt = "You are an expert in mathematics and Lean 4."
    informal_problem = "The real numbers $x, y, z$ satisfy $0 \\leq x \\leq y \\leq z \\leq 4$. If their squares form an arithmetic progression with common difference 2, determine the minimum possible value of $|x-y|+|y-z|$.\n Prove that the answer is: 4-2\\sqrt{3}"
    header = "import Mathlib\n\nopen Real\n"
    ground_truth = "theorem omni_theorem_637 :\n    IsLeast {n : ℝ | ∃ x y z, 0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 4 ∧ y^2 - x^2 = 2 ∧ z^2 - y^2 = 2 ∧ n = abs (x - y) + abs (y - z)} (4 - 2 * sqrt 3) := by\n  sorry"
    user_prompt = utils.get_formal_statement_prompt(informal_problem, header)

    dialog = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ] 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True) + "<think>"
    print(f"prompt: {prompt}")

    model = LLM(
        MODEL_DIR, 
        tensor_parallel_size=4 # 8 for 32B, 4 for 7B
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        n=16
    )

    responses = model.generate(prompt, sampling_params)
    formal_statements = []
    for i, output in enumerate(responses[0].outputs):
        response_text = output.text
        formal_statements.append(utils.post_process_formal_statement_with_reasoning(response_text, header))
        if i == 0:
            print(f"response {i}: {response_text}")

    leanClient = utils.KiminaLeanClient(
        host        = "0.0.0.0",
        port        = 12332,
        timeout     = 60,
        max_workers = 32
    )

    header = header.strip("\n")
    p2q_code_list = []
    q2p_code_list = []
    ground_truth  = utils.replace_theorem_name(ground_truth, "thm_Q")
    for i in range(len(formal_statements)):
        formal_statements[i] = utils.replace_theorem_name(formal_statements[i], "thm_P")
        p2q_code = utils.get_p2q_proof_by_exact(formal_statements[i], ground_truth, header)
        q2p_code = utils.get_p2q_proof_by_exact(ground_truth, formal_statements[i], header)
        p2q_code_list.append(p2q_code)
        q2p_code_list.append(q2p_code)

    p2q_results = leanClient.query(code_list=p2q_code_list)
    q2p_results = leanClient.query(code_list=q2p_code_list)
    results     = [
        {
            "p2q_pass": utils.judge_beq_correctness(p2q, formal_statement),
            "q2p_pass": utils.judge_beq_correctness(q2p, ground_truth),
            "p2q_result": p2q,
            "q2p_result": q2p,
        }
        for formal_statement, p2q, q2p in zip(formal_statements, p2q_results, q2p_results)
    ]
    num_passed = 0
    for result in results:
        result["beq_pass"] = result["p2q_pass"] and result["q2p_pass"]
        if result["beq_pass"]:
            num_passed += 1

    print(f"Pass: {num_passed} / {len(results)}")



