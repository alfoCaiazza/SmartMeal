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
EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" 

def validate_paths(path):
    if not path.exists():
        raise FileNotFoundError(f"Data or file not founded: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Directory not founded: {path.parent}")

def embedding_builder():
    try:
        validate_paths(DATA_PATH)
        validate_paths(EMBEDDING_PATH)
        
        logger.info("Loading dataset...")
        try:
            df = pd.read_csv(DATA_PATH)
            if df.empty:
                raise ValueError("Loaded empy DataFrame")
        except Exception as e:
            logger.error(f"ERROR in loading dataset: {str(e)}")
            raise

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("recipe-embedding-generation")

        with mlflow.start_run(run_name="Embeddings Model Generator"):
            # Impostazione del dispositivo (GPU/CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Running on: {device}")
            
            logger.info("Loading SentenceTransformer model...")
            try:
                model = SentenceTransformer("all-mpnet-base-v2", device=device)
            except Exception as e:
                logger.error(f"ERROR in loading model: {str(e)}")
                raise

            texts = df['EmbeddingDescription'].astype(str).tolist()

            # Creazione degli embeddings
            logger.info("Embeddings generation...")
            embeddings = []
            batch_size = 1024

            for i in tqdm(range(0, len(texts), batch_size), desc="Recipes Embeddings"):
                batch = texts[i:i+batch_size]
                emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
                embeddings.extend(emb)

            # Saving embeddings
            logger.info(f"Saving embeddings in {EMBEDDING_PATH}")
            try:
                np.save(EMBEDDING_PATH, embeddings)
                logger.info("Successfully saved embeddings")
            except Exception as e:
                logger.error(f"ERROR in saving embeddings: {str(e)}")
                raise

            mlflow.log_artifact(EMBEDDING_PATH, artifact_path="recipes_embeddings")
    except Exception as e:
        logger.error(f"CRITIC ERROR during embedding_builder execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    embedding_builder()