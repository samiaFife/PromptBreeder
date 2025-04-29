import contextlib
import gc
import json
import os
import random

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM

from src.evaluation.evaluator import GenerationEvaluator, TextClassificationEvaluator
from src.utils.data import INNER_GENERATION_TASKS
from src.utils.load_dataset import load_dataset


class ModelLoader:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._args = args
            cls._instance._kwargs = kwargs
        return cls._instance

    def __init__(self, task_name: str = None, bench_name=None, batch_size=None):
        if not self._initialized:
            self.task_name = task_name
            self.bench_name = bench_name
            self._batch_size = batch_size
            self._model = None
            self._tokenizer = None
            self._terminators = None
            self._device = None
            self._evaluator = None
            self._max_new_tokens = None
            self._labels = None
            self._base_prompt = None
            self.initialize()
            self.__class__._initialized = True
        else:
            print("Model already initialized, skipping...")

    def initialize(self):
        if self._initialized:
            print("Model already initialized, skipping...")
            return

        print("Starting model initialization...")
        self._initialized = True

        self.seed_everything(42)
        torch.cuda.empty_cache()
        gc.collect()

        # *self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # *device_map = "auto" if self._device == "cuda" else None

        model_name = "AnatoliiPotapov/T-lite-instruct-0.1"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self._model = LLM(model=model_name, dtype=torch.float16, trust_remote_code=True)

        # *print(f"Model loaded on {self._device}")
        print("Model loaded via vllm")
        # *if device_map is None:
        # *    self._model = self._model.to(self._device)

        # *self._model.eval()

        print("Model eval completed")

        if self.task_name in ["gsm8k", "math", "samsum"] or self.task_name in INNER_GENERATION_TASKS:
            self._evaluator = GenerationEvaluator()
            self._max_new_tokens = 100
        else:
            self._evaluator = TextClassificationEvaluator()
            self._max_new_tokens = 50
        print(f"Evaluator loaded: {self._evaluator}")

        with open("../../../data/labels.json", "r") as f:
            labels_json = json.load(f)
            try:
                if self.bench_name is not None:
                    self._labels = labels_json[self.bench_name][self.task_name]
                else:
                    self._labels = labels_json[self.task_name]
            except KeyError:
                self._labels = []

        with open("../../../data/basic_prompts.json", "r") as f:
            prompts_json = json.load(f)
            if self.bench_name is not None:
                self._base_prompt = prompts_json[self.bench_name][self.task_name]
            else:
                self._base_prompt = prompts_json[self.task_name]

        print("Model initializing completed")
        self.print_gpu_memory()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_generate_args_hf(self):
        return {"max_new_tokens": self._max_new_tokens, "eos_token_id": self._terminators}

    @property
    def model_generate_args(self):
        return {"max_tokens": self._max_new_tokens, "stop_token_ids": self._terminators, "temperature": 0.75}

    @property
    def device(self):
        return self._device

    def load_data(self, prompt, split="train", sample=100):
        return load_dataset(
            self.task_name,
            tokenizer=self._tokenizer,
            sample=sample if split == "train" else None,
            split=split,
            prompt=prompt,
            device=self._device,
        )

    @property
    def labels(self):
        return self._labels

    @property
    def base_prompt(self):
        return self._base_prompt

    @property
    def evaluator(self):
        return self._evaluator

    def seed_everything(self, seed: int = 42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        # np.random.seed(seed)
        np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def print_gpu_memory(self):
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            print(
                f"GPU {i}: Used = {used/1024**3:.2f} GB | Free = {free/1024**3:.2f} GB | Total = {total/1024**3:.2f} GB"
            )

    def get_metrics(self, candidate, split="train"):
        return self.evaluator.evaluate_vllm(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=self.load_data(candidate, split),
            batch_size=self.batch_size,
            model_generate_args=self.model_generate_args,
        )

    def generate(self, prompt):
        return self.model.generate(prompt, self.model_generate_args)

    def get_sample(self):
        return self.load_data(sample=1)

    def destroy(self):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        del self._model.llm_engine.model_executor
        del self._model
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
