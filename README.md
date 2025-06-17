# Tool Hallucination

> This is the official implementation of our ICML 2024 paper [Reducing Tool Hallucination via Reliability Alignment](https://arxiv.org/abs/2412.04141). The data and code will be publicly available soon.

# ⚙️ Setting up the Virtual API Server

Our **RelyToolBench** tool environment is fully built on top of the **StableToolBench** Virtual API Server. The **only difference** is that we simulate tool responses using `gpt-4o-2024-08-06` instead of `gpt-4-turbo-2024-04-09`, due to its lower cost.

To set up the server, please refer to the [StableToolBench repository](https://github.com/THUNLP-MT/StableToolBench). The basic steps are as follows:

1. **Apply for ToolBench keys**.
2. **Configure the Virtual API Server**.  
   You can customize variables in `server/config`, such as `api_key` and `api_base`, to fit your environment.  
   If you're using Docker, we recommend building the server with our updated Dockerfile at `server/Dockerfile-new`. The original Dockerfile from StableToolBench had some version mismatches, which may or may not be fixed in their latest release.

---

# 📂 Queries in RelyToolBench

The queries are located in the `solvable_queries/test_instruction/` folder.  
Files suffixed with `_missing_all_arguments` and `_with_unmatched_tools` are **synthetically generated** by us.  

To reduce evaluation cost, we only construct **unsolvable queries** for the following three subsets:

- `G1_instruction`
- `G2_category`
- `G3_instruction`

Please refer to our paper for detailed synthesis methodology.

---

# 🚀 Inference on RelyToolBench

If you haven't set up the environment yet, please run:

```
pip install -r requirements.txt
```

We primarily use **[vLLM](https://github.com/vllm-project/vllm)** for inference. The main script is:

```bash
scripts_eval/inference_toolllama_vllm_split.sh
```

Run it with:

```bash
bash scripts_eval/inference_toolllama_vllm_split.sh $PORT $MODEL_NAME $MODEL_PATH
```

Where:

- `PORT`: The port number for the API Server.
- `MODEL_NAME`: A string that must include the model type (e.g., `llama3`, `qwen`, or `toolllama`) to help identify the backbone.
- `MODEL_PATH`: Path to your model checkpoint.

⚠️ Make sure to update `SERVICE_URL` and `VLLM_API_BASE` in the script according to your local setup.

This script performs two tasks:
1. Launches 8 independent vLLM servers (one per GPU) on an 8-GPU machine.
2. Launches the multi-threaded evaluation script: `toolbench/inference/qa_pipeline_multithread.py`.

If your setup uses a different GPU configuration or multiple machines, you’ll need to modify the script accordingly.

This evaluation script `toolbench/inference/qa_pipeline_multithread.py` reads settings from a config file. You can either modify `config.yml` directly or provide your own path. Required fields include:

```yaml
api_key: 
api_base: 
toolbench_key: 
tool_root_dir: 
```

---

### 🧪 Example: Single-GPU Inference Script

To run a single-GPU inference on a specific subset, use:

```bash
export PYTHONPATH=./
export VLLM_API_BASE=""   # Address of vllm server
export SERVICE_URL=""     # Address of API server
export MODEL_PATH="llama3"
export STRATEGY="CoT@1"

export OUTPUT_DIR="data_test/answer/virtual_llama3-1_cot"
group="G1_instruction_with_unmatched_tools"

mkdir -p $OUTPUT_DIR/$group

python toolbench/inference/qa_pipeline_multithread.py \
    --backbone_model llama3 \
    --model_path ${MODEL_PATH} \
    --max_observation_length 1024 \
    --method ${STRATEGY} \
    --input_query_file solvable_queries/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --max_query_count 30 \
    --num_thread 1
```

---

# 📊 Evaluation on RelyToolBench

We follow the evaluation protocol from **StableToolBench**, with one key **extension**:  
We evaluate **tool hallucinations** and compute a **Reliable Pass Rate (RePR)**, which excludes failures due to tool hallucination.

## Step 1: Prepare Prediction Files

This is consistent with the ToolEval workflow from StableToolBench.  
Prepare a directory to store model predictions, organized by test set. Then run the following script to convert the format:

```bash
cd toolbench/tooleval

MODEL_NAME=$1
CANDIDATE_MODEL=$2
export RAW_ANSWER_PATH=../../data_eval/answer_${MODEL_NAME}
export CONVERTED_ANSWER_PATH=../../data_eval/model_predictions_converted_${MODEL_NAME}

test_sets=(
    "G1_instruction_missing_all_arguments"
    "G1_instruction_with_unmatched_tools"
    "G2_category"
    "G2_category_missing_all_arguments"
    "G2_category_with_unmatched_tools"
    "G3_instruction"
    "G3_instruction_missing_all_arguments"
    "G3_instruction_with_unmatched_tools"
    "G1_instruction"
)

for test_set in "${test_sets[@]}"; do
    mkdir -p ${CONVERTED_ANSWER_PATH}/${CANDIDATE_MODEL}/
    answer_dir=${RAW_ANSWER_PATH}/${CANDIDATE_MODEL}/${test_set}
    output_file=${CONVERTED_ANSWER_PATH}/${CANDIDATE_MODEL}/${test_set}.json

    python convert_to_answer_format.py \
        --answer_dir ${answer_dir} \
        --method CoT@1 \
        --output ${output_file}
done
```

- `MODEL_NAME`: Name of the top-level directory (e.g., `answer_chatgpt_cot`)
- `CANDIDATE_MODEL`: Sub-directory name for the specific model

## Step 2: Evaluate Metrics

Run the following script to compute:

- Pass Rate
- Tool Hallucination Rate
- Reliable Pass Rate (RePR)

```bash
bash run_pass_rate.sh
```

⚠️ The `run_pass_rate.sh` script depends on OpenAI's services, so you need to provide the address of the OpenAI service in the `openai_key.json` file.

---

# Relign Alignment Algorithm

Our Relign algorithm optimizes model alignment for tool hallucination, consisting of two main steps: SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization).

During SFT, we synthesize training data to teach the model how to refuse tool calls under abnormal conditions—such as when the tool is incompatible with the task or the user query lacks necessary arguments. However, we found that the model's performance after SFT alone was unsatisfactory.

To further improve alignment, we apply DPO. Specifically, we sample outputs from the SFT-trained model on a subset of tasks, and then use our hallucination evaluator to rank these outputs (those without hallucinations are preferred over those with hallucinations). Based on these rankings, we construct DPO training pairs to optimize the model.

Both our SFT and DPO data are derived from samples in the ToolBench training set. The SFT data is statically synthesized from ToolBench's training set, while the DPO data is dynamically generated through sampling and evaluation using our framework.

The SFT data file is `toolllama_train_sft.json`. You can fine-tune your model on this dataset using Hugging Face’s `trl` library.

The DPO sampling tasks are located in the `solvable_queries/test_instruction` folder under the files:

* `G1_train_4k_idx1.json`
* `G1_train_4k_idx2.json`
* `G1_train_4k_idx3.json`

These tasks are sampled from ToolLLaMA's training data and are disjoint from those used in SFT.

To perform sampling, use the script `scripts_eval/sampling_toolllama_vllm_split.sh`. Compared to the regular inference script, the key difference is that it enables the `--do_sampling` flag in the `toolbench/inference/qa_pipeline_multithread.py` function, allowing the model to sample multiple outputs at each step. These outputs are stored in the `sampling_outputs` field of the inference result.

After obtaining multi-sample outputs for each step of the task, you can use the notebook `generate_sampling_dpo_data.ipynb` to generate DPO training data. This involves using a hallucination evaluator to classify the outputs and construct DPO data pairs accordingly.

---

# 📚 Citation

If you find this benchmark useful in your work, please cite:

```bibtex
@article{xu2024reducing,
  title={Reducing tool hallucination via reliability alignment},
  author={Xu, Hongshen and Zhu, Zichen and Pan, Lei and Wang, Zihan and Zhu, Su and Ma, Da and Cao, Ruisheng and Chen, Lu and Yu, Kai},
  journal={arXiv preprint arXiv:2412.04141},
  year={2024}
}
```
