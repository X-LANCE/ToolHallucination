cd toolbench/tooleval

MODEL_NAME=$1
CANDIDATE_MODEL=$2
export RAW_ANSWER_PATH=../../data_eval/answer_${MODEL_NAME}
export CONVERTED_ANSWER_PATH=../../data_eval/model_predictions_converted_${MODEL_NAME}


test_sets=("G1_instruction_missing_all_arguments" "G1_instruction_with_unmatched_tools" "G2_category" "G2_category_missing_all_arguments" "G2_category_with_unmatched_tools" "G3_instruction" "G3_instruction_missing_all_arguments" "G3_instruction_with_unmatched_tools" "G1_instruction")

for test_set in "${test_sets[@]}"; do
    mkdir -p ${CONVERTED_ANSWER_PATH}/${CANDIDATE_MODEL}/
    answer_dir=${RAW_ANSWER_PATH}/${CANDIDATE_MODEL}/${test_set}
    output_file=${CONVERTED_ANSWER_PATH}/${CANDIDATE_MODEL}/${test_set}.json

    python convert_to_answer_format.py \
        --answer_dir ${answer_dir} \
        --method CoT@1 \
        --output ${output_file}
done
