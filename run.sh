#!/bin/bash
echo "running task: PromptBreeder"
TASKS=("sst-2")
META_DIR=("./logs/test/$TASK")
for TASK in "${TASKS[@]}"; do
    echo "running task $TASK"
    python ./main.py \
        --task-name $TASK \
        --meta-dir $META_DIR
    echo "finished task $TASK"
done