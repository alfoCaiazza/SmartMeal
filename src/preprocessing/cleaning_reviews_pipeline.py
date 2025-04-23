import mlflow.artifacts
import pandas as pd
import numpy as np
import mlflow
import logging
from functools import reduce
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
INPUT_PATH = ROOT_PATH / "data"/"raw"/"reviews.csv"
OUTPUT_PATH = ROOT_PATH/"data"/"processed"

##
# Cleaning Reviews Pipeline Steps
# STEP 1: Handling pesudo-date object and expanding them into 3 new features: year, month and day
# STEP 2: Dropping useless featues
# STEP 3: Dropping data duplicates, preserving only one sample
# STEP 4: Labeling the dataset (0,1) based on the value of each rating
# #

# Logging config
logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

# STEP 1:
def decompose_data_object_data(df:pd.DataFrame, col_date: str = 'DateSubmitted') -> pd.DataFrame:
    try:
        logging.info("Starting data objects decomposition")
        df[col_date] = pd.to_datetime(df[col_date], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
        df['Year'] = df[col_date].dt.year
        df['Month'] = df[col_date].dt.month
        df['Day'] = df[col_date].dt.day
        logging.info("Decomposition successully completed!")
    except Exception as e:
        logging.error(f"ERROR in date decomposition: {e}")

    return df

# STEP 2:
def drop_features(df: pd.DataFrame,feature_list: list, ) -> pd.DataFrame:
    try:
        logging.info("Dropping feature list from specified dataframe")
        df = df.drop(columns=feature_list)
        logging.info("Dropping action completed!")
    except Exception as e:
        logging.info(f"ERROR in dropping feature: {e}")

    return df

# STEP 3:
def delete_duplicates(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    initial_rows = df.shape[0]
    df = df.drop_duplicates(subset=subset, keep='first')
    final_rows = df.shape[0]
    logging.info(f"Duplicates removed: {initial_rows - final_rows}")
    return df

# STEP 4:
def label_dataset(df: pd.DataFrame, rating_col: str = 'Rating') -> pd.DataFrame:
    logging.info("Starting dataset labeling")
    df['Label'] = np.where(df[rating_col] >= 3, 1, 0)
    logging.info("Labeling completed")
    return df

# Pipeline composition
def pipeline(df:pd.DataFrame, steps: list) -> pd.DataFrame:
    return reduce(lambda data, func: func(data), steps, df)

def cleaning_rev_pipeline():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("review-cleaning-pipeline")

    logging.info(f"Loading dataset from path: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    useless_features = [
        'DateModified',
        'DateSubmitted',
        'Review',
        'AuthorName'
    ]

    steps = [
        lambda df: decompose_data_object_data(df, col_date='DateSubmitted'),
        lambda df: drop_features(df, feature_list=useless_features),
        lambda df: delete_duplicates(df, subset=['ReviewId', 'AuthorId']),
        lambda df: label_dataset(df, rating_col='Rating')
    ]

    with mlflow.start_run(run_name="Cleaning Reviews Dataset"):
        mlflow.log_param("input_path", str(INPUT_PATH))
        mlflow.log_param("output_path", str(OUTPUT_PATH))
        logging.info("Starting pipeline ... ")
        df_processed = pipeline(df, steps)

        output_file = OUTPUT_PATH / "cleaned_reviews.csv"
        df_processed.to_csv(output_file, index=False)

        mlflow.log_artifact(str(output_file), artifact_path="processed_review")
        logging.info(f"Clean dataset saved in : {output_file}")

if __name__ == "__main__":
    cleaning_rev_pipeline()