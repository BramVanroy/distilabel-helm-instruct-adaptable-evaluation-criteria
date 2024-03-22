import json
import os
from pathlib import Path

from datasets import load_dataset
from distilabel.dataset import DatasetCheckpoint
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline

from helm_instruct.criterion.base import Criterion, Rating
from helm_instruct.criterion.nl import default_criterion
from helm_instruct.evaluator.evaluator import HelmInstructTask
from helm_instruct.evaluator.template.nl import template

from openai import AzureOpenAI

# To support Azure, install this branch: https://github.com/BramVanroy/distilabel/tree/support-azure-openai
# pip install git+https://github.com/BramVanroy/distilabel.git@support-azure-openai

HF_API_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HF_AUTH_TOKEN")
DATASET_NAME = "BramVanroy/ultra_feedback_dutch_cleaned"
NEW_DATASET_NAME = "BramVanroy/ultra_feedback_dutch_cleaned_rated"
MODELS_OF_INTEREST = ["GEITje-7B-ultra", "gpt-4-turbo"]
SAVE_FREQUENCY = 1000

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.rename_column("prompt", "prompt_english")
dataset = dataset.rename_column("prompt_dutch", "prompt")

system_prompt_dutch = "Je bent een automatische annotator die de kwaliteit van de tekst van een AI-model beoordeelt aan de hand van gegeven criteria. De tekst van het AI-model is een reactie op een gegeven instructie en moet die instructie dus goed beantwoorden of volgen."
task_description_dutch = "Het volgende is een instructie geschreven door een mens (`Instructie:`), en een reactie op de instructie geschreven door een AI-model (`Reactie:`). Beantwoord alstublieft de volgende vragen over de reactie van het AI-model, rekening houdend met de gegeven opties (`Opties:`)."

default_criterion["Dutch-ness"] = Criterion(
    question="Is de tekst in vlot en gramaticaal correct Nederlands geschreven? Negeer code-fragmenten in je analyse en richt je enkel op de doorlopende tekst. Leenwoorden uit andere talen mogen gebruikt worden als dat gewoonlijk is in het domein (bv. bij software).",
    ratings=[
        Rating(
            value=1,
            description="De tekst is onleesbaar of bevat veel grammaticale fouten.",
        ),
        Rating(
            value=2,
            description="De tekst is moeilijk te begrijpen of bevat veel grammaticale fouten.",
        ),
        Rating(
            value=3,
            description="De tekst is begrijpelijk maar bevat enkele grammaticale fouten.",
        ),
        Rating(
            value=4,
            description="De tekst is goed geschreven en bevat weinig grammaticale fouten.",
        ),
        Rating(
            value=5,
            description="De tekst is uitstekend geschreven, vlot leesbaar en bevat geen grammaticale fouten.",
        ),
    ],
)
del default_criterion["Harmlessness"]
del default_criterion["Understandability"]
del default_criterion["Completeness"]

checkpoint_strategy = DatasetCheckpoint(
    strategy="hf-hub",
    extra_kwargs={
        "repo_id": NEW_DATASET_NAME,
        "private": False,
        "split": "train",
    },
    save_frequency=SAVE_FREQUENCY,
)

# Expects a JSON file that has as main keys "profiles", e.g. "gpt-4-turbo"
# and as sub-keys "api_key", "api_version", "azure_endpoint", "azure_deployment"
credentials = json.loads(Path(".credentials.json").read_text(encoding="utf-8"))["gpt-4-turbo"]

azure_client = AzureOpenAI(
    api_key=credentials["api_key"],
    api_version=credentials["api_version"],
    azure_endpoint=credentials["azure_endpoint"],
)

for column_name in MODELS_OF_INTEREST:
    skip_dry_run = False
    dataset = dataset.rename_column(
        column_name, "response"
    )  # set column to correct input column
    for criterion_key, criterion_value in default_criterion.items():
        pipe = Pipeline(
            labeller=OpenAILLM(
                model=credentials["azure_deployment"],
                client=azure_client,
                task=HelmInstructTask(
                    template=template,
                    criterion=criterion_value,
                    system_prompt=system_prompt_dutch,
                    task_description=task_description_dutch,
                ),
                max_new_tokens=10,
                num_threads=8,
                temperature=0,
            )
        )
        dataset = pipe.generate(
            dataset,
            batch_size=100,
            skip_dry_run=skip_dry_run,
            checkpoint_strategy=checkpoint_strategy,
        )
        # rename columns to avoid overwriting data
        criterion_column = f"{criterion_key}_{column_name}"

        dataset = dataset.rename_column("rating", f"rating_{criterion_column}")
        skip_dry_run = True
    # convert back to original column name to avoid losing data
    dataset = dataset.rename_column("response", column_name)

dataset.push_to_hub(NEW_DATASET_NAME, token=HF_API_TOKEN)
