#!/bin/bash
echo "running task: PromptBreeder"
TASKS_BBH=("boolean_expressions" "hyperbaton" "temporal_sequences" "object_counting" "disambiguation_qa" "logical_deduction_three_objects" "logical_deduction_five_objects" "logical_deduction_seven_objects" "causal_judgement" "date_understanding" "ruin_names" "word_sorting" "geometric_shapes" "movie_recommendation" "salient_translation_error_detection" "formal_fallacies" "penguins_in_a_table" "dyck_languages" "multistep_arithmetic_two" "navigate" "reasoning_about_colored_objects" "tracking_shuffled_objects_three_objects" "tracking_shuffled_objects_five_objects" "tracking_shuffled_objects_seven_objects" "sports_understanding" "snarks" "web_of_lies")
TASKS=("gsm8k" "math" "medqa" "mnli" "mr" "openbookqa" "qnli" "samsum" "sst-2" "trec" "yahoo")
TASKS_NAT_INSTR=("task021" "task050" "task069")
for TASK in "${TASKS[@]}"; do
    echo "running task $TASK"
    META_DIR="./logs/test/$TASK"
    META_NAME="$TASK-1.txt"
    META_DIR_TEST="./logs/test/metrics.txt"
    mkdir -p $META_DIR
    python ./main.py \
        --task-name $TASK \
        --meta-dir $META_DIR \
        --meta-name $META_NAME \
        --meta-dir-test $META_DIR_TEST
    echo "finished task $TASK"
done
for TASK in "${TASKS_NAT_INSTR[@]}"; do
    echo "running task $TASK"
    META_DIR="./logs/test/$TASK"
    META_NAME="$TASK-1.txt"
    META_DIR_TEST="./logs/test/metrics.txt"
    mkdir -p $META_DIR
    python ./main.py \
        --task-name $TASK \
        --meta-dir $META_DIR \
        --meta-name $META_NAME \
        --meta-dir-test $META_DIR_TEST \
        --bench-name "natural_instructions"
    echo "finished task $TASK"
done
for TASK in "${TASKS_BBH[@]}"; do
    echo "running task $TASK"
    META_DIR="./logs/test/$TASK"
    META_NAME="$TASK-1.txt"
    META_DIR_TEST="./logs/test/metrics.txt"
    mkdir -p $META_DIR
    python ./main.py \
        --task-name $TASK \
        --meta-dir $META_DIR \
        --meta-name $META_NAME \
        --meta-dir-test $META_DIR_TEST \
        --bench-name "bbh"
    echo "finished task $TASK"
done