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
    Calculate the mean embedding of the recipes appreciated by a specified user,
    referring to a .npy file as the embedding source and a recipe ID mapping.
    """

    logger.info(f"Loading embeddings from: {embedding_path}")
    logger.info(f"Loading ID mapping from: {id_map_path}")

    try:
        embeddings = np.load(embedding_path)
        recipe_id_map = np.load(id_map_path)
        id_to_index = {rid: idx for idx, rid in enumerate(recipe_id_map)}
    except Exception as e:
        logger.error(f"ERROR loading npy files: {e}")
        raise

    label = 1
    recipe_ids = get_user_history(df, user_id, label)

    if len(recipe_ids) == 0:
        logger.warning(f"No liked recipes found for user {user_id}")
        raise ValueError(f"User {user_id} has no liked recipes.")

    try:
        embeddings_subset = np.array([
            embeddings[id_to_index[rid]] for rid in recipe_ids if rid in id_to_index
        ])
    except KeyError as e:
        logger.error(f"Some RecipeId not found in global embedding map: {e}")
        raise

    avg_embedding = np.mean(embeddings_subset, axis=0)
    logger.info(f"Mean embedding calculated for {embeddings_subset.shape[0]} liked recipes")

    return avg_embedding

def build_user_embedding(df: pd.DataFrame, embedding_path: Path, id_map_path: Path, output_dir: Path):
    """
    Calculates and saves avg mean profile for each user in the reffered dataframe.
    """
    logger.info("Building user profile embeddings...")

    user_ids = []
    user_embeddings = []

    for user_id in df["AuthorId"].unique():
        try:
            avg_embedding = get_user_average_embedding_from_file(df, user_id, embedding_path, id_map_path)
            user_ids.append(user_id)
            user_embeddings.append(avg_embedding)
        except Exception as e:
            logger.warning(f"Skipping user {user_id}: {e}")

    user_ids = np.array(user_ids)
    user_embeddings = np.array(user_embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "user_ids.npy", user_ids)
    np.save(output_dir / "user_profiles.npy", user_embeddings)

    logger.info(f"Saved {len(user_ids)} user profiles to: {output_dir}")


if __name__ == "__main__":
    ROOT_PATH = Path(__file__).parent.parent
    INPUT_PATH = ROOT_PATH / "data" / "processed" / "cleaned_reviews.csv"
    EMBEDDING_PATH = ROOT_PATH / "data" / "processed" / "embeddings" / "recipe_embeddings.npy"
    ID_MAPPING_PATH = ROOT_PATH / "data" / "processed" / "embeddings" / "recipe_ids.npy"
    OUTPUT_USER_PROFILE_DIR = ROOT_PATH / "data" / "processed" / "embeddings"

    validate_paths(INPUT_PATH)
    validate_paths(EMBEDDING_PATH )
    validate_paths(ID_MAPPING_PATH)
    validate_paths(OUTPUT_USER_PROFILE_DIR)

    df = load_dataset(INPUT_PATH)

    build_user_embedding(df, EMBEDDING_PATH, ID_MAPPING_PATH, OUTPUT_USER_PROFILE_DIR)





