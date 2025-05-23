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
STEP 4: filling missing values 
STEP 5: standardizing ingredients
STEP 6: extracting allergens from recipe ingredients
STEP 7: creates a unified field for feature embedding
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
    
# STEP 4
def fill_null_fields(df, fields) -> pd.DataFrame:
    """
    Replases null values with empty string considering specified columns
    """
    df_copy = df.copy()

    for field in fields:
        df_copy[field] = df_copy[field].fillna("")

    return df_copy

# STEP 5
def standardize_ingredients(elem):
    """
    Deduplicates and normalizes ingredents inside a list
    """
    if isinstance(elem, list):
        cleaned = set()
        for ing in elem:
            ing = ing.lower()
            ing = re.sub(r'[^a-z0-9 ]', '', ing)  # rimuove simboli
            ing = re.sub(r'\s+', ' ', ing).strip()  # spazi multipli
            cleaned.add(ing)
        return sorted(list(cleaned))
    return elem

# STEP 6
ALLERGENS_DICT = {
    "milk": ["milk", "cheese", "butter", "cream", "yogurt"],
    "egg": ["egg", "eggs"],
    "nuts": ["almond", "pecan", "walnut", "cashew", "hazelnut", "nut"],
    "soy": ["soy", "soy sauce", "tofu"],
    "gluten": ["flour", "wheat", "bread", "cracker", "pasta", "biscuit", "cake", "noodle"],
    "fish": ["fish", "salmon", "tuna"],
    "shellfish": ["shrimp", "lobster", "crab", "clam", "mussel"]
}

def extract_allergens(ingredients):
    if isinstance(ingredients, list):
        found = set()
        for allergen, keywords in ALLERGENS_DICT.items():
            for ing in ingredients:
                if any(kw in ing for kw in keywords):
                    found.add(allergen)
        return sorted(list(found))
    return []

# STEP 7

def join_features(df, ft1, ft2, new_ft) -> pd.DataFrame:
    """
    Merges two columns into a new unified one. If entries are lists it does a joint with ","
    """

    def join_row(row):
        f1 = ", ".join(row[ft1]) if isinstance(row[ft1], list) else str(row[ft1])
        f2 = ", ".join(row[ft2]) if isinstance(row[ft2], list) else str(row[ft2])

        return f"{f1} , {f2}"
    
    df_copy = df.copy()
    df_copy[new_ft] = df_copy.apply(join_row, axis=1)

    return df_copy

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

        # Selecting only columns of interest
        usefull_cols = [
            'RecipeId',
            'Name',
            'RecipeCategory',
            'Keywords',
            'RecipeIngredientParts'
        ]
        df = df[usefull_cols]

        # Categorizing features
        pseudo_list_cols = ['Keywords', 'RecipeIngredientParts']
        other_text_cols = ['Name', 'RecipeCategory']

        for col in pseudo_list_cols:
            df[col] = df[col].apply(clean_pseudo_list)

        # Defining pipeline functions (STEP 1, STEP 2, STEP 3)
        text_cleaner = compose_pipeline(
            clean_pseudo_list,
            remove_escapes,
            remove_punctuation,
        )

        print("Initializing and starting cleaning pipeline:")
        for col in tqdm(other_text_cols, desc="Cleaning columns"):
            df[col] = df[col].apply(text_cleaner)

        # Fill missing values (STEP 4)
        nulll_features = df.columns[df.isnull().any()].tolist()
        fill_null_fields(df, nulll_features)

        # Processing data standardization and extracion (STEP 5)
        print("Extracting recipes ingredients")
        df["RecipeIngredientParts"] = df["RecipeIngredientParts"].apply(standardize_ingredients)

        # STEP 6
        print("Extracting recipes allergens") 
        df["Allergens"] = df["RecipeIngredientParts"].apply(extract_allergens)
        mlflow.log_metric("recipes_with_allergens", df["Allergens"].apply(bool).sum())

        # STEP 7
        print("Creating full recipes description")
        df = join_features(df, 'Keywords', 'RecipeIngredientParts', 'EmbeddingDescription')

        df.to_csv(output_path, index=False)

        mlflow.log_artifact(str(output_path), artifact_path="processed_data")
        mlflow.log_metric("final_rows", len(df))

        print(f"Run completed and tracked with MLflow!")

if __name__ == "__main__":
    input_csv = INPUT_PATH / "recipes.csv"
    output_csv = OUTPUT_PATH / "cleaned_recipes.csv"

    cleaning_pipeline(input_csv, output_csv)