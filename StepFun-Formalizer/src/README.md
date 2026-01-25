# Inference and Evaluation Code of Stepfun-Formalizer

We use [vllm](https://github.com/vllm-project/vllm) for the generation of formal statements. For the verification of Lean 4 code, we modify [kimina-lean-server](https://github.com/wyt2000/kimina-lean-server) to prevent exceeding memory limits. For evaluation, we integrate the strictest BEq check (see Definition 3.2 and Appendix A.5 in paper) into our code.

## Setup

### Clone the Repo and Submodules
```bash
git clone --recurse-submodules https://github.com/stepfun-ai/StepFun-Formalizer
```

### Install Requirements

#### vllm

```bash
pip install -r requirements.txt
```

#### Kimina Lean Server

Follow the instructions of the submodule `kimina-lean-server`.

### Start Lean Server

```bash
bash start_kimina_lean_server.sh
```

## Inference and Evaluation

Make sure the `MODEL_DIR` and `RESULT_PATH` variables are correctly set in the scripts.

### See the Reasoning Trajectory of Stepfun-Formalizer

```bash
python eval.py
```

### Evaluating the Performance of StepFun-Formalizer on a JSONL Dataset

```bash
python eval_benchmarks.py
```

The format of the dataset is shown in `datasets/formallite_combibench_proverbench.jsonl`.
