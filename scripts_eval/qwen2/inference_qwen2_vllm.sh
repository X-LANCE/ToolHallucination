export PYTHONPATH=./
export VLLM_API_BASE="http://127.0.0.1:8310/v1/"   # the address of vllm.server
export SERVICE_URL="http://d8-hpc-gpu-006:8080/virtual"  # the address of api server
export MODEL_PATH="qwen2" # the name of vllm.server
export STRATEGY="CoT@1"  # or CoT@1

export OUTPUT_DIR="data_eval/answer_test/virtual_qwen2_cot"  # change it accordingly

group=G1_instruction  # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python toolbench/inference/qa_pipeline_multithread.py \
    --backbone_model qwen2 \
    --model_path ${MODEL_PATH} \
    --max_observation_length 1024 \
    --method ${STRATEGY} \
    --input_query_file solvable_queries/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --max_query_count 30 \
    --num_thread 1