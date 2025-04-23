import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import torch
import mlflow
from tqdm import tqdm
import sys
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_recipes.csv"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

def validate_paths(path):
    if not path.exists():
        raise FileNotFoundError(f"Data or file not founded: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Directory not founded: {path.parent}")

def embedding_builder():
    try:
        validate_paths(DATA_PATH)
        validate_paths(EMBEDDING_DIR)
        
        logger.info("Loading dataset...")
        try:
            df = pd.read_csv(DATA_PATH)
            if df.empty:
                raise ValueError("Loaded empy DataFrame")
        except Exception as e:
            logger.error(f"ERROR in loading dataset: {str(e)}")
            raise

        recipe_ids = df['RecipeId'].values
        texts = df['EmbeddingDescription'].astype(str).tolist()

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("recipe-embedding-generation")

        with mlflow.start_run(run_name="Embeddings Model Generator"):
            # Impostazione del dispositivo (GPU/CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Running on: {device}")
            
            logger.info("Loading SentenceTransformer model...")
            try:
                model = SentenceTransformer(MODEL_NAME, device=device)
            except Exception as e:
                logger.error(f"ERROR in loading model: {str(e)}")
                raise

            # Embeddings generation
            logger.info("Embeddings generation...")
            embeddings = []

            embeddings = model.encode(texts, batch_size=1024, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

            # Convert to numpy array
            embeddings = np.array(embeddings)

            # Save to file
            emb_file = EMBEDDING_DIR / "recipe_embeddings.npy"
            id_file = EMBEDDING_DIR / "recipe_ids.npy"

            np.save(emb_file, embeddings)
            np.save(id_file, recipe_ids)

            logger.info(f"Saved embeddings to {emb_file}")
            logger.info(f"Saved recipe ID mapping to {id_file}")

            mlflow.log_param("embedding_model", MODEL_NAME)
            mlflow.log_metric("num_embeddings", embeddings.shape[0])
            mlflow.log_artifact(str(emb_file), artifact_path="embeddings")
            mlflow.log_artifact(str(id_file), artifact_path="embeddings")
    except Exception as e:
        logger.error(f"CRITIC ERROR during embedding_builder execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    embedding_builder()