#!/bin/bash
echo "running task: PromptBreeder"
TASKS=("sst-2")
META_DIR=("./logs/test/$TASK")
META_NAME="$TASK-1.txt"
for TASK in "${TASKS[@]}"; do
    echo "running task $TASK"
    python ./main.py \
        --task-name $TASK \
        --meta-dir $META_DIR \
        --meta-name $META_NAME
    echo "finished task $TASK"
done