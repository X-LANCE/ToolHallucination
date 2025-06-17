export PYTHONPATH=./
#export VLLM_API_BASE="http://127.0.0.1:8083/v1/"  # the address of vllm.server
export SERVICE_URL="http://d8-hpc-gpu-006:8080/virtual" # the address of api server
#export MODEL_PATH="toolllama"  # the name of vllm.server
#export STRATEGY="DFS_woFilter_w2"  # or CoT@1
export STRATEGY="CoT@1"

MODEL_NAME=$1
MODEL_PATH=$2


if [[ "$MODEL_NAME" == *"llama3"* ]]; then
    BACKBONE_MODEL="llama3"
elif [[ "$MODEL_NAME" == *"qwen2"* ]]; then
    BACKBONE_MODEL="qwen2"
elif [[ "$MODEL_NAME" == *"toolllama"* ]]; then
    BACKBONE_MODEL="ToolLLaMA_vllm"
else
    echo "Error: MODEL_NAME does not contain a valid identifier (toolllama, llama3, qwen2)."
    exit 1
fi

export OUTPUT_DIR="data_eval/answer/$MODEL_NAME"  # change it accordingly

function wait_for_vllm_ready {
    local port=$1
    echo "Waiting for vLLM on port $port to be ready..."
    while true; do
        if curl -s http://localhost:$port/v1/models | grep "id" > /dev/null; then
            echo "vLLM on port $port is ready!"
            break
        else
            sleep 2
        fi
    done
}

BASE_PORT=8510
BASE_VLLM_PORT=5330
for i in $(seq 0 7); do
    PORT=$(($BASE_PORT + $i))
    VLLM_PORT_PER_DEVICE=$(($BASE_VLLM_PORT + $i))
    echo "Starting vLLM on GPU $i, port $PORT..., vllm port $VLLM_PORT_PER_DEVICE"
    VLLM_PORT=$VLLM_PORT_PER_DEVICE CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
          --model $MODEL_PATH \
          --served-model-name ${BACKBONE_MODEL} $ --max-model-len=8192 --dtype=bfloat16 --disable-frontend-multiprocessing \
          --host 127.0.0.1 --port $PORT --rope-scaling '{"rope_type": "linear", "factor": 2.0}' > log/vllm_gpu_$i.log 2>&1 &
done

for i in $(seq 0 7); do
    PORT=$(($BASE_PORT + $i))
    wait_for_vllm_ready $PORT
    #echo "vLLM on port $PORT is ready."
done


#group=G1_instruction  # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction
#mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
test_sets=("G1_instruction" "G1_instruction_missing_all_arguments" "G1_instruction_with_unmatched_tools" "G2_category" "G2_category_missing_all_arguments" "G2_category_with_unmatched_tools" "G3_instruction" "G3_instruction_missing_all_arguments" "G3_instruction_with_unmatched_tools")

for test_set in "${test_sets[@]}"
do
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$test_set
done

for round in $(seq 1 3); do
echo "Starting round $round..."

PIDS=()
for i in $(seq 0 7); do
    PORT=$(($BASE_PORT + $i))
    VLLM_API_BASE="http://127.0.0.1:$PORT/v1/" python toolbench/inference/qa_pipeline_multithread.py \
    --backbone_model ${BACKBONE_MODEL} \
    --model_path ${BACKBONE_MODEL} \
    --max_observation_length 1024 \
    --method ${STRATEGY} \
    --input_query_file solvable_queries/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR \
    --max_query_count 30 \
    --num_thread 4 \
    --test_sets "${test_sets[@]}" \
    --task_split_num 8 \
    --task_split_index $i > log/task_gpu_$i.log 2>&1 &

    PIDS+=($!)
done

for PID in "${PIDS[@]}"; do
    wait $PID
done

done
echo "All rounds have completed."


pkill -9 -f "vllm.entrypoints.openai.api_server"