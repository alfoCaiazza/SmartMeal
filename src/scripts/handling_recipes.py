import pandas as pd
from pathlib import Path
import sys
import logging
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_recipes.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed.npy"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_paths(path):
    if not path.exists():
        raise FileNotFoundError(f"Data or file not founded: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Directory not founded: {path.parent}")

def load_dataset(path:str) -> pd.DataFrame:
    validate_paths(path)
        
    logger.info("Loading dataset...")
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Loaded empy DataFrame")
    except Exception as e:
        logger.error(f"ERROR in loading dataset: {str(e)}")
        raise

    return df

def load_embeddings(path:str) -> np.ndarray:
    validate_paths(path)
        
    logger.info("Loading recipe embeddings...")
    try:
        embeddings = np.load(path)
    except Exception as e:
        logger.error(f"ERROR in loading embeddings: {str(e)}")
        raise

    return embeddings

def get_recipe_embedding(df: pd.DataFrame, embeddings: np.ndarray, recipe_id: int) -> np.ndarray:
    """
    Returns the embedding of the recipe with the given RecipeId.
    Assumes embeddings[i] corrisponde a df.iloc[i].
    """
    logger.info("Searching recipe ...")
    try:
        row = df[df['RecipeId'] == recipe_id]
        
        if row.empty:
            raise ValueError(f"No recipe found with RecipeId = {recipe_id}")
        
       
        index = row.index[0] 
        embedding = embeddings[index]

    except Exception as e:
        logger.error(f"ERROR in finding recipe embedding: {str(e)}")
        raise

    logger.info("Embedding found.")
    print("Max index in embeddings:", embeddings.shape[0] - 1)
    print("RecipeId richiesto:", recipe_id)
    print("Posizione nel DataFrame:", df[df['RecipeId'] == recipe_id].index)

    return embedding


if __name__ == "__main__":
    df = load_dataset(INPUT_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    output = get_recipe_embedding(df, embeddings, 111111)
    print(output)