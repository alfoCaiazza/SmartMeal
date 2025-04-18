import os
import zipfile
import mlflow

def unzip_dataset(target_dir, zip_filename):
    zip_path = os.path.join(target_dir, zip_filename)
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print('Dataset downloaded and unzipped correctly.')

        # Log downloaded dataset as an artifact
        mlflow.log_artifact(zip_path, artifact_path="raw_dataset_zip")
    else:
        print('ERROR: zip file not found')
        mlflow.log_param("status", "FAILED")

def download_dataset():
    print('Starting dataset downloading ...')

    # MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("dataset-downloads")

    with mlflow.start_run(run_name="Download Food.com Dataset"):
        #Logged Params
        dataset_name = "foodcom-recipes-and-reviews"
        target_dir = "src/data/raw"
        zip_filename = f"{dataset_name}.zip"

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("download_dir", target_dir)

        os.makedirs(target_dir, exist_ok=True)

        # Downloading dataset via CLI
        os.system(f'kaggle datasets download -d irkaal/{dataset_name} -p {target_dir}')

        unzip_dataset(target_dir, zip_filename)

if __name__ == "__main__":
    download_dataset()
