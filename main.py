import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from rich import print

load_dotenv()  # load environment variables

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run the PromptBreeder Algorithm. Number of units is mp * ts.")
parser.add_argument("-mp", "--num_mutation_prompts", type=int, default=3)
parser.add_argument("-ts", "--num_thinking_styles", type=int, default=4)
parser.add_argument("-e", "--num_evals", type=int, default=100)
parser.add_argument("-n", "--simulations", type=int, default=10)
parser.add_argument("--task-name", default="sst-2")
parser.add_argument("--bench-name", default="")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--meta-dir")
parser.add_argument("--meta-name")
parser.add_argument("--meta-dir-test", help="Directory for test results in JSON format")
parser.add_argument("--eval-only", type=bool, default=False)

args = parser.parse_args()

logger.info(f"args: {args}")

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from pb.utils.model_loader import ModelLoader  # noqa 402
logger.info("ModelLoader imported")
from pb.mutation_prompts import mutation_prompts  # noqa 402
from pb.thinking_styles import thinking_styles  # noqa 402
from pb import create_population, init_run, run_for_n  # noqa 402

logger.info("Imports executed")

if args.bench_name != "":
    loader = ModelLoader(task_name=args.task_name, bench_name=args.bench_name, batch_size=args.batch_size)
else:
    loader = ModelLoader(task_name=args.task_name, batch_size=args.batch_size)
loader.seed_everything()
logger.info("Model loaded")

if args.eval_only:
    output_dir = os.path.dirname(args.meta_dir_test)
    output_file_path = os.path.join(output_dir, "plum_metrics_full.txt")

    with open(output_file_path, "a") as output_file:
        with open(args.meta_dir_test, "r") as input_file:
            for line in input_file:
                try:
                    data = json.loads(line.strip())
                    task_name = data.get("task")
                    prompt = data.get("prompt")

                    loader.initialize_task(task_name)
                    if prompt == "":
                        prompt = loader.base_prompt
                    logger.info(f"Starting scoring {task_name}, evaluator: {loader.evaluator}, prompt: {prompt}")
                    score = loader.get_metrics(candidate=prompt, split="test", full=True)

                    result = {"task_name": task_name, "score": score, "prompt": prompt}
                    logger.info(result)
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()
                    logger.info(f"Processed prompt for task: {task_name}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse line as JSON: {line}")
                except Exception as e:
                    logger.error(f"Error processing line: {str(e)}")

    logger.info(f"Evaluation completed. Results written to {output_file_path}")
    loader.destroy()
    sys.exit(0)

meta_path = os.path.join(args.meta_dir, args.meta_name)
meta_file = open(meta_path, "w+")
meta_test_path = open(args.meta_dir_test, "a")

num_samples = 100
total_evaluations = args.num_mutation_prompts * args.num_thinking_styles * num_samples

# // # set num_workers to total_evaluations so we always have a thread
# // co = cohere.Client(
# //     api_key=os.environ["COHERE_API_KEY"], num_workers=total_evaluations, max_retries=5, timeout=30
# // )  # override the 2 min timeout with 30s.

tp_set = mutation_prompts[: int(args.num_mutation_prompts)]
mutator_set = thinking_styles[: int(args.num_thinking_styles)]

logger.info(f"You are prompt-optimizing for the problem: {loader.base_prompt}")

logger.info("Creating the population...")
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=loader.base_prompt)

logger.info("Generating the initial prompts...")
init_run(p, loader, num_samples)

logger.info("Starting the genetic algorithm...")
run_for_n(n=int(args.simulations), population=p, loader=loader)

print("%" * 80)
print("done processing! final gen:")
print(p.units)

max_fitness = max(unit.fitness for unit in p.units)
fittest = [(unit.fitness, unit.P) for unit in p.units if unit.fitness == max_fitness]

meta_file.write(f"initital prompt: {loader.base_prompt}\nthe fittest:\n" + "-" * 80 + "\n")
for fitness, prompt in fittest:
    meta_file.write(f"fitness: {fitness}\n")
    meta_file.write(f"prompt: {prompt}\n")
    meta_file.write("-" * 80 + "\n")

test_score = loader.get_metrics(candidate=fittest[0][1], split="test", full=False)
result = {"task_name": args.task_name, "score": test_score, "prompt": fittest[0][1]}
meta_test_path.write(json.dumps(result) + "\n")

loader.destroy()
