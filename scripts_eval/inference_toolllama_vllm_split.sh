#!/bin/bash
pkill -9 -f "vllm.entrypoints.openai.api_server"
export PYTHONPATH=./
#export VLLM_API_BASE="http://127.0.0.1:8083/v1/"  # the address of vllm.server
#export MODEL_PATH="toolllama"  # the name of vllm.server
#export STRATEGY="DFS_woFilter_w2"  # or CoT@1
export STRATEGY="CoT@1"

SERVICE_PORT=$1
MODEL_NAME=$2
MODEL_PATH=$3

export SERVICE_URL="http://d8-hpc-gpu-006:$SERVICE_PORT/virtual" # the address of api server


test_sets=("G1_instruction" "G1_instruction_missing_all_arguments" "G1_instruction_with_unmatched_tools" "G2_category" "G2_category_missing_all_arguments" "G2_category_with_unmatched_tools" "G3_instruction" "G3_instruction_missing_all_arguments" "G3_instruction_with_unmatched_tools")
#test_sets=("G1_instruction" "G2_category" "G3_instruction")
#test_sets=("G1_instruction" "G1_instruction_with_unmatched_tools" "G1_instruction_missing_all_arguments")

if [[ "$MODEL_NAME" == *"llama3"* ]]; then
    BACKBONE_MODEL="llama3"
elif [[ "$MODEL_NAME" == *"qwen"* ]]; then
    BACKBONE_MODEL="qwen2"
elif [[ "$MODEL_NAME" == *"toolllama"* ]]; then
    BACKBONE_MODEL="ToolLLaMA_vllm"
else
    echo "Error: MODEL_NAME does not contain a valid identifier (toolllama, llama3, qwen2)."
    exit 1
fi

export OUTPUT_DIR="data_eval/answer_$BACKBONE_MODEL/$MODEL_NAME"  # change it accordingly
function get_recursive_file_count {
    find "$1" -type f | wc -l
}

for test_set in "${test_sets[@]}"
do
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$test_set
done
mkdir -p log/$MODEL_NAME

initial_count=$(get_recursive_file_count "$OUTPUT_DIR")
if [[ "$initial_count" -ge 950 ]]; then
    echo "文件夹下的文件数量大于或等于 950，程序退出。"
    exit 0
fi

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

BASE_PORTS=(11330 11430 11530 11630 11730 11830 11930 12030)
BASE_VLLM_PORTS=(21330 21430 21530 21630 21730 21830 21930 22030)

# 获取端口列表的长度
num_ports=${#BASE_PORTS[@]}

# 随机选择一个端口
random_index=$((RANDOM % num_ports))

BASE_PORT=${BASE_PORTS[$random_index]}
BASE_VLLM_PORT=${BASE_VLLM_PORTS[$random_index]}

#BASE_PORT=8510
#BASE_VLLM_PORT=5330
for i in $(seq 0 7); do
    PORT=$(($BASE_PORT + $i))
    VLLM_PORT_PER_DEVICE=$(($BASE_VLLM_PORT + $i))
    echo "Starting vLLM on GPU $i, port $PORT..., vllm port $VLLM_PORT_PER_DEVICE"
    if [[ "$BACKBONE_MODEL" == "ToolLLaMA_vllm" ]]; then
      VLLM_PORT=$VLLM_PORT_PER_DEVICE CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
          --model $MODEL_PATH \
          --served-model-name ${BACKBONE_MODEL} $ --max-model-len=8192 --dtype=bfloat16 --disable-frontend-multiprocessing \
          --host 127.0.0.1 --port $PORT --rope-scaling '{"rope_type": "linear", "factor": 2.0}' > log/$MODEL_NAME/vllm_gpu_$i.log 2>&1 &
    else
      VLLM_PORT=$VLLM_PORT_PER_DEVICE CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
          --model $MODEL_PATH \
          --served-model-name ${BACKBONE_MODEL} $ --max-model-len=8192 --dtype=bfloat16 --disable-frontend-multiprocessing \
          --host 127.0.0.1 --port $PORT > log/$MODEL_NAME/vllm_gpu_$i.log 2>&1 &
    fi
done

for i in $(seq 0 7); do
    PORT=$(($BASE_PORT + $i))
    wait_for_vllm_ready $PORT
    #echo "vLLM on port $PORT is ready."
done


#group=G1_instruction  # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction
#mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

TIMEOUTS=(1 3 5 7 9)
TIMEOUTS=(5 7 9 10 12 15)

for iteration in $(seq 1 6); do
echo "Starting iteration $iteration..."
initial_count=$(get_recursive_file_count "$OUTPUT_DIR")
echo "Initial file count of $OUTPUT_DIR: $initial_count"


random_seed=$RANDOM
PIDS=()
process_indices=()
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
    --num_thread 8 \
    --test_sets "${test_sets[@]}" \
    --task_split_num 8 \
    --random_seed $random_seed \
    --task_split_index $i > log/$MODEL_NAME/task_gpu_$i.log 2>&1 &

    PIDS+=($!)
    process_indices+=($i)
done

wait_time=${TIMEOUTS[$((iteration-1))]}
all_processes_terminated=false
sleep 30

while true; do

    current_count=$(get_recursive_file_count "$OUTPUT_DIR")
    for ((j=1; j<=wait_time; j++)); do
      echo "  wait  $j-th 10sec of $wait_time"
      sleep 10
      for i in "${!PIDS[@]}"; do
          pid="${PIDS[$i]}"
          if ! kill -0 $pid &>/dev/null; then
              #echo "PIDS: ${PIDS[@]}"
              echo "remained_process_indices: ${process_indices[@]}"
              #echo "process ${process_indices[$i]} (PID: $pid) has terminated."
              PIDS=(${PIDS[@]/$pid})
              process_indices=(${process_indices[@]/${process_indices[$i]}})
          fi
      done
      if [[ ${#PIDS[@]} -eq 0 ]]; then
        all_processes_terminated=true
      fi
      if $all_processes_terminated; then
          echo "All process finished, go to iteration $((iteration + 1))"
          break
      fi
    done

    new_count=$(get_recursive_file_count "$OUTPUT_DIR")
    if [[ "$new_count" -eq "$current_count" ]]; then
        echo "File count $current_count has not changed during $wait_time seconds, go to iteration $((iteration + 1))"
        for pid in "${PIDS[@]}"; do
            kill $pid
        done
        break
    fi

done
echo "All iterations have completed."
done

pkill -9 -f "vllm.entrypoints.openai.api_server"
#sleep 150