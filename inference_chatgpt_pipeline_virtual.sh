export TOOLBENCH_KEY="N3VJwMatu0jqBmSceRhxI9fsFdLHo5Xp6vK4QgrEiO8kyY2A1C"

export OPENAI_KEY="sk-Rc8kklUxKwikzsTWAcD182Fc384647D3A5141cDa78B691E6"
export OPENAI_API_BASE="https://api.xi-ai.cn/v1"
export PYTHONPATH=./
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

GPT_MODEL=$1
#export GPT_MODEL="gpt-4o-2024-08-06"
export SERVICE_URL="http://d8-hpc-gpu-006:8080/virtual"

export OUTPUT_DIR="data_eval/answer_apimodel/$GPT_MODEL"
category=test_instruction
#group=G1_instruction_missing_all_arguments
#mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

test_sets=("G1_instruction" "G1_instruction_missing_all_arguments" "G1_instruction_with_unmatched_tools" "G2_category" "G2_category_missing_all_arguments" "G2_category_with_unmatched_tools" "G3_instruction" "G3_instruction_missing_all_arguments" "G3_instruction_with_unmatched_tools")
test_sets=("G3_instruction_missing_all_arguments" "G3_instruction_with_unmatched_tools")

for test_set in "${test_sets[@]}"
do
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$test_set
done


python toolbench/inference/qa_pipeline_multithread.py \
    --backbone_model chatgpt_function \
    --chatgpt_model $GPT_MODEL \
    --max_observation_length 1024 \
    --method CoT@1 \
    --output_answer_file $OUTPUT_DIR/$group \
    --test_sets "${test_sets[@]}" \
    --num_thread 50
