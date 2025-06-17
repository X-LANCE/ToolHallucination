#!/bin/bash
MODEL_NAME=$1
CANDIDATE_MODEL=$2

bash run_convert_answer_single.sh ${MODEL_NAME} ${CANDIDATE_MODEL}

cd  toolbench/tooleval
export API_POOL_FILE=../../openai_key.json

export EVAL_MODEL=gpt-4o-mini
#export EVAL_MODEL=gpt-4-turbo-2024-04-09
export CONVERTED_ANSWER_PATH=../../data_eval/model_predictions_converted_${MODEL_NAME}
#export SAVE_PATH=../../data_eval/pass_rate_results_${MODEL_NAME}_${EVAL_MODEL}
export SAVE_PATH=../../data_eval/pass_rate_results_${MODEL_NAME}
#export EVAL_MODEL=gpt-4o-mini-2024-07-18
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

test_sets=("G1_instruction" "G1_instruction_with_unmatched_tools" "G1_instruction_missing_all_arguments" "G2_category" "G2_category_missing_all_arguments" "G2_category_with_unmatched_tools" "G3_instruction" "G3_instruction_missing_all_arguments" "G3_instruction_with_unmatched_tools")


python eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids ../../solvable_queries/test_query_ids \
    --max_eval_threads 50 \
    --evaluate_times 1 \
    --test_set "${test_sets[@]}"
