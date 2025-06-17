cd  toolbench/tooleval

MODEL_NAME=$1
CANDIDATE_MODEL=$2

export CONVERTED_ANSWER_PATH=../../data_eval/model_predictions_converted_${MODEL_NAME}
export SAVE_PATH=../../data_eval/orig_pass_rate_results_${MODEL_NAME}
mkdir -p ${SAVE_PATH}
#export CANDIDATE_MODEL="virtual_llama3-1_dfs" # change it accordingly
export EVAL_MODEL=gpt-4o-2024-08-06
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

export test_set=G1_instruction # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction

python orig_eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids ../../solvable_queries/test_query_ids \
    --max_eval_threads 1 \
    --evaluate_times 3 \
    --test_set ${test_set} \
    # --overwrite