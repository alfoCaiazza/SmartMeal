import pandas as pd
from pathlib import Path
import sys
import logging
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

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

# def load_embeddings(path:str) -> np.ndarray:
#     validate_paths(path)
        
#     logger.info("Loading recipe embeddings...")
#     try:
#         embeddings = np.load(path)
#     except Exception as e:
#         logger.error(f"ERROR in loading embeddings: {str(e)}")
#         raise

#     return embeddings

def get_user_history(df: pd.DataFrame, user_id: int) -> np.ndarray:
    """
    Returns the embedding of the recipe with the given RecipeId.
    Assumes embeddings[i] corrisponde a df.iloc[i].
    """
    logger.info(f"Searching recipe  for user id : {user_id}")
    try:
        reviews_history = df['RecipeId'].loc[df['AuthorId'] == user_id]
        
        if reviews_history.empty:
            raise ValueError(f"No reviews history found for user = {user_id}")
    except Exception as e:
        logger.error(f"ERROR in finding recipe embedding: {str(e)}")
        raise

    return reviews_history


if __name__ == "__main__":
    df = load_dataset(INPUT_PATH)
    # embeddings = load_embeddings(EMBEDDINGS_PATH)
    output = get_user_history(df, 2046)
    print(f"Total reviews retrived: {len(output)}\n")
    print(output)