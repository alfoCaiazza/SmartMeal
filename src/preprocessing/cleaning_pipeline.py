from functools import reduce
from typing import Callable
import string
import mlflow
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm

"""
Defining a data cleaning pipeline in order to get the recipes ready for the AI model
STEP 1: removing 'c()' pattern from specific feature
STEP 2: removing escape and non alfanumerical chars
STEP 3: removing punctation chars
"""

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH =  PROJECT_ROOT / "data" / "processed"

# STEP 1
def clean_pseudo_list(elem: str) -> list | str:
    """
    Converts a stringa with a pattern like 'c("item1", "item2", ...)' 
    into a list.
    """
    if not isinstance(elem, str):
        return elem

    # Finds element that matches regex pattern: specifically words between marks
    matches = re.findall(r'"([^"]*)"', elem)

    return [item.strip() for item in matches if item.strip()] if matches else elem

# STEP 2
def remove_escapes(elem) -> str:
    """
    Removes escape char (\n, \t), then converts string in lower case.
    """
    def clean_string(s: str) -> str:
        s = re.sub(r'[\n\t]', ' ', s)
        s = re.sub(r'[^a-zA-Z0-9 \-]', '', s)
        return s.lower()
    
    if isinstance(elem, list):
        return [clean_string(s) for s in elem] #if isinstance(s, str)
    elif isinstance(elem, str):
        return clean_string(elem)
    else:
        return elem


# STEP 3
def remove_punctuation(elem) -> str:
    """
    Removes punctuation marks in a string or a string list
    """

    translator = str.maketrans('','', string.punctuation)

    if isinstance(elem, list):
        return [s.translate(translator) for s in elem if isinstance(s, str)]
    elif isinstance(elem, str):
        return elem.translate(translator)
    else:
        return elem

def compose_pipeline(*functions: Callable) -> Callable:
    """
    Combines a functions sequence into a single pipeline function
    """

    return lambda x: reduce(lambda acc, f: f(acc), functions, x)

def cleaning_pipeline(input_path, output_path):
    # MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("data-cleaning-pipeline")

    with mlflow.start_run(run_name="Cleaning textual features"):
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("output_path", str(output_path))

        print(f"Loading dataset from path: {input_path}")
        df = pd.read_csv(input_path)
        print("Dataset loaded successfully!")
        mlflow.log_metric("initial_rows", len(df))

        # Defining pipeline functions
        text_cleaner = compose_pipeline(
            clean_pseudo_list,
            remove_escapes,
            remove_punctuation
        )

        # Selecting only columns of interest
        usefull_cols = [
            'RecipeId',
            'Name',
            'Description',
            'RecipeCategory',
            'Keywords',
            'RecipeIngredientParts'
        ]
        df = df[usefull_cols]

        # Selectiong textual columns
        text_columns = df.select_dtypes(include=['object']).columns
        mlflow.log_param("text_columns", ",".join(text_columns))

        print("Initializing and starting cleaning pipeline:")
        for col in tqdm(text_columns, desc="Cleaning columns"):
            df[col] = df[col].apply(text_cleaner)

        df.to_csv(output_path, index=False)

        mlflow.log_artifact(str(output_path), artifact_path="processed_data")
        mlflow.log_metric("final_rows", len(df))

        print(f"Run completed and tracked with MLflow!")

if __name__ == "__main__":
    input_csv = INPUT_PATH / "recipes.csv"
    output_csv = OUTPUT_PATH / "cleaned_recipes.csv"

    cleaning_pipeline(input_csv, output_csv)