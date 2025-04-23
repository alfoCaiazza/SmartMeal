import pandas as pd
from pathlib import Path
import logging
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_paths(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Data or file not founded: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Directory not founded: {path.parent}")

def load_dataset(path: Path) -> pd.DataFrame:
    validate_paths(path)
    
    logger.info(f"Loading dataset from {path}")
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Loaded empty DataFrame")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    return df

def get_user_history(df: pd.DataFrame, user_id: int, label: int) -> np.ndarray:
    """
    Retrives the review ids and the recipes ids of the reviews made by a specified user
    """
    logger.info(f"Retrieving recipe history for user ID: {user_id}")
    reviews_history = df.loc[(df['AuthorId'] == user_id) & (df['Label'] == label), 'RecipeId'].values
    
    if len(reviews_history) == 0:
        error_msg = f"No review history found for user ID {user_id}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return reviews_history

if __name__ == "__main__":
    df = load_dataset(INPUT_PATH)
    user_id = 2046
    label = 0
    recipes_reviewed = get_user_history(df, user_id, label)

    logger.info(f"Total reviews retrieved for user {user_id}: {len(recipes_reviewed)}")
    print(f"Recipe IDs:\n{recipes_reviewed}")
