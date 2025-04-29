import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles
from rich import print

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from pb.utils.model_loader import ModelLoader  # noqa 402

load_dotenv()  # load environment variables

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run the PromptBreeder Algorithm. Number of units is mp * ts.")
parser.add_argument("-mp", "--num_mutation_prompts", type=int, default=2)
parser.add_argument("-ts", "--num_thinking_styles", type=int, default=4)
parser.add_argument("-e", "--num_evals", type=int, default=100)
parser.add_argument("-n", "--simulations", type=int, default=10)
parser.add_argument("--task-name", default="sst-2")
parser.add_argument("--bench-name", default="")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("-p", "--problem", default="Solve the math word problem, giving your answer as an arabic numeral.")

args = vars(parser.parse_args())

if args.bench_name != "":
    loader = ModelLoader(task_name=args.task_name, bench_name=args.bench_name, batch_size=args.batch_size)
else:
    loader = ModelLoader(task_name=args.task_name, batch_size=args.batch_size)
loader.seed_everything()
logger.info("Model loaded")

num_samples = 100
total_evaluations = args["num_mutation_prompts"] * args["num_thinking_styles"] * num_samples

# // # set num_workers to total_evaluations so we always have a thread
# // co = cohere.Client(
# //     api_key=os.environ["COHERE_API_KEY"], num_workers=total_evaluations, max_retries=5, timeout=30
# // )  # override the 2 min timeout with 30s.

tp_set = mutation_prompts[: int(args["num_mutation_prompts"])]
mutator_set = thinking_styles[: int(args["num_thinking_styles"])]

logger.info(f"You are prompt-optimizing for the problem: {loader.base_prompt}")

logger.info("Creating the population...")
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=loader.base_prompt)

logger.info("Generating the initial prompts...")
init_run(p, loader, num_samples)

logger.info("Starting the genetic algorithm...")
run_for_n(n=int(args["simulations"]), population=p, model=loader, num_evals=num_samples)

print("%" * 80)
print("done processing! final gen:")
print(p.units)

loader.destroy()
