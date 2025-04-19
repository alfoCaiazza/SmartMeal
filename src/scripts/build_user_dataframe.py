import pandas as pd
import mlflow
from tqdm import tqdm
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REVIEWS_PATH = INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

def build_user_profiles():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("user-profile-generation")

    with mlflow.start_run(run_name="Simulated User Profiles"):
        reviews = pd.read_csv(REVIEWS_PATH)

        # Filtering users based on the number of reviews
        most_active_users = reviews['AuthorId'].value_counts()
        most_active_users = most_active_users[most_active_users > 50].index.tolist()
        sample_users = random.sample(most_active_users, k=min(2000, len(most_active_users)))

        # Possibile preferences, intolerances and cuisines
        possible_prefs = [
            "vegetarian", "vegan", "low-carb", "keto", "gluten-free", "paleo",
            "high-protein", "low-fat", "dairy-free", "whole30"
        ]
        possible_intol = [
            "gluten", "lactose", "nuts", "egg", "soy", "fish", "shellfish"
        ]
        possible_cuisines = [
            "italian", "indian", "japanese", "mediterranean", "american",
            "thai", "chinese", "mexican", "french", "nordic"
        ]

        # Creating users
        user_profiles = {}

        for user_id in sample_users:
            user_profiles[user_id] = {
                "UserId": user_id,
                "Preferences": random.sample(possible_prefs, k=random.randint(1,3)),
                "Intolerances": random.sample(possible_intol, k=random.randint(0,2)),
                "Cuisines": random.sample(possible_cuisines, k=random.randint(1,3))
            }

        df_profiles = pd.DataFrame.from_dict(user_profiles, orient="index")
        os.makedirs("src/data/processed", exist_ok=True)
        output_path = "src/data/processed/user_profiles.csv"
        df_profiles.to_csv(output_path, index=False)

        mlflow.log_artifact(output_path, artifact_path="user_profiles")
        print(f"User profiles saved in: {output_path}")

if __name__ == "__main__":
    build_user_profiles()



        