import pandas as pd
import numpy as np
from pathlib import Path
from handlig_user import get_user_history, validate_paths, load_dataset
import logging


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_user_average_embedding_from_file(df: pd.DataFrame, user_id: int, embedding_path: Path, id_map_path: Path) -> np.ndarray:
    """
    Calculate the mean embedding of the recipes appreciated by a specified user, reffering to a .npy file as the embedding source
    """

    logger.info(f"Loading embeddings from: {embedding_path}")
    logger.info(f"Loading ID mapping from: {id_map_path}")

    try:
        embeddings = np.load(embedding_path)
    except Exception as e:
        logger.error(f"ERROR in loading embeddings: {e}")
        raise

    # Creating mapping RecipeId -> index
    label = 1
    recipe_ids = get_user_history(df, user_id, label)
    id_to_index = {rid: idx for idx, rid in enumerate(recipe_ids)}
    logger.info(f"Calculating mean embedding for user_id {user_id} considering liked recipes")

    if len(recipe_ids) == 0:
        logger.warning(f"No liked recipes found for user {user_id}")
        raise ValueError(f"User {user_id} has no liked recipes.")

    #Retrievs embedding using the mapping
    try:
        embeddings_subset = np.array([embeddings[id_to_index[rid]] for rid in recipe_ids if rid in id_to_index])
    except KeyError as e:
        logger.error(f"Some RecipeId not found in embedding map: {e}")
        raise

    avg_embedding = np.mean(embeddings_subset, axis=0)
    logger.info(f"Mean embedding calulated for {embeddings_subset.shape[0]} liked recipes")

    return avg_embedding

if __name__ == "__main__":
    ROOT_PATH = Path(__file__).parent.parent
    INPUT_PATH = ROOT_PATH / "data" / "processed" / "cleaned_reviews.csv"
    EMBEDDING_PATH  = ROOT_PATH / "data" / "processed" / "embeddings" / "recipe_embeddings.npy"
    ID_MAPPING_PATH = ROOT_PATH / "data" / "processed" / "embeddings" / "recipe_ids.npy"

    validate_paths(INPUT_PATH)
    validate_paths(EMBEDDING_PATH )
    validate_paths(ID_MAPPING_PATH)

    df = load_dataset(INPUT_PATH)
    user_id = 2625

    avg_embedding = get_user_average_embedding_from_file(df, user_id, EMBEDDING_PATH, ID_MAPPING_PATH)
    print(f"Mean embedding: {avg_embedding}")




