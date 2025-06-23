# src/run_benchmark.py

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate  # you can swap this out for your own solvers
from inspect_ai.scorer import mean, stderr, exact
from inspect_ai.solver import solver as make_solver

#––– YOUR SCORERS (aliases for brevity) ––––––––––––––––––––––––––––––––––––––––––––––––
# you’d import & configure these exactly as you had before:
from inspect_ai.scorer import bleurt_20, model_graded_qa_custom, trace_score, rule_coverage, explainability_fidelity

#––– YOUR SOLVERS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# wrap your solvers in a list so they conform to the example’s `[generate()]` form
# here we alias `vanilla_generate(...)` to just `vanilla` for brevity
from inspect_ai.solver import solver as make_solver
from your_module import vanilla_generate, rag_kg_nesy_solver  # wherever you defined them

@task
def benchmark_vanilla():
    """Benchmark vanilla GPT-4o-mini on the diabetes prompts."""
    return Task(
        dataset=json_dataset("diabetes_prompts.jsonl"),
        solver=[generate()],
        scorer=[
            model_graded_qa(),
            f1,
            trace_score(),
            rule_coverage(),
            explainability_fidelity(),
        ],
    )

# Uncomment once your RAG+KG solver is ready:
# @task
# def benchmark_rag_kg():
#     """Benchmark RAG+KG decision support system."""
#     return Task(
#         dataset=json_dataset("diabetes_prompts.jsonl"),
#         solver=[rag_kg_nesy_solver(config_path="config.yaml")],
#         scorer=[
#             bleurt_20(),
#             model_graded_qa_custom(),
#             trace_score(),
#             rule_coverage(),
#             explainability_fidelity(),
#         ],
#         attribs={"name": "rag_kg_nesy"},
#     )
