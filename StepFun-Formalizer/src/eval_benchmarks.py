from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import tqdm
from collections import defaultdict, Counter
import numpy as np

import eval_utils as utils 

MODEL_DIR    = "/path/to/Stepfun-Formalizer/"
DATASET_PATH = "datasets/formallite_combibench_proverbench.jsonl"
RESULT_PATH  = "/path/to/save/evaluation/results/"
NUM_SAMPLES  = 16

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = LLM(
        MODEL_DIR, 
        tensor_parallel_size=4 # 7B
#        tensor_parallel_size=8 # 32B
    )

    dataset = []
    dataset_names = defaultdict(set)
    with open(DATASET_PATH) as f:
        for data in f:
            data = json.loads(data)
            dataset.append(data)
            dataset_names[data["benchmark_name"]].add(data["name"])

    system_prompt = "You are an expert in mathematics and Lean 4."

    generation_results = []
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for data in dataset:
            user_prompt = utils.get_formal_statement_prompt(data["problem"], data["header"])

            dialog = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ] 

            prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True) + "<think>"

            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=16384,
                n=NUM_SAMPLES
            )

            responses = model.generate(prompt, sampling_params, use_tqdm=False)
            for i, output in enumerate(responses[0].outputs):
                response_text = output.text
                generation_results.append(
                    data | {
                        "raw_response": response_text,
                        "formal_statement": utils.post_process_formal_statement_with_reasoning(response_text, data["header"]),
                    }
                )
            pbar.update(1)
        
    leanClient = utils.KiminaLeanClient(
        host        = "0.0.0.0",
        port        = 12332,
        timeout     = 60,
        max_workers = 100
    )

    p2q_code_list = []
    q2p_code_list = []
    for generation in generation_results:
        generation["formal_statement"] = utils.replace_theorem_name(generation["formal_statement"], "thm_P")
        generation["ground_truth"] = utils.replace_theorem_name(generation["ground_truth"], "thm_Q")
        p2q_code = utils.get_p2q_proof_by_exact(generation["formal_statement"], generation["ground_truth"], generation["header"])
        q2p_code = utils.get_p2q_proof_by_exact(generation["ground_truth"], generation["formal_statement"], generation["header"])
        p2q_code_list.append(p2q_code)
        q2p_code_list.append(q2p_code)

    p2q_results = leanClient.query(code_list=p2q_code_list, is_clear_cache=True)
    q2p_results = leanClient.query(code_list=q2p_code_list, is_clear_cache=True)
    verification_results = [
        data | {
            "p2q_pass": utils.judge_beq_correctness(p2q, data["formal_statement"]),
            "q2p_pass": utils.judge_beq_correctness(q2p, data["ground_truth"]),
            "p2q_result": p2q,
            "q2p_result": q2p,
        }
        for data, p2q, q2p in zip(generation_results, p2q_results, q2p_results)
    ]

    passed_counter = defaultdict(Counter)
    with open(RESULT_PATH, "w") as f:
        for result in verification_results:
            result["beq_pass"] = result["p2q_pass"] and result["q2p_pass"]
            if result["beq_pass"]:
                passed_counter[result["benchmark_name"]][result["name"]] += 1
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    for benchmark, names in dataset_names.items():
        pass_1 = []
        pass_k = []
        for name in names:
            pass_1.append(utils.pass_at_k(NUM_SAMPLES, passed_counter[benchmark][name], 1))
            pass_k.append(utils.pass_at_k(NUM_SAMPLES, passed_counter[benchmark][name], NUM_SAMPLES))
        print(f"[{benchmark}] Pass@1: {np.mean(pass_1):.3%}")
        print(f"[{benchmark}] Pass@{NUM_SAMPLES}: {np.mean(pass_k):.3%}")



