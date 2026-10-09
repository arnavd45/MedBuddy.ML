import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# loading .env content
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIRS") / os.getenv("LOG_NAME_1")

logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

# load the trained model only once (module-level cache)
model = load(MODEL_PATH)
logging.info("Model loaded sucessfully")

def predict(input_data:dict):
    df = pd.DataFrame([input_data])
    # get predicted class
    prediction = int(model.predict(df)[0])
    # get prediction probability
    probability = float(model.predict_proba(df)[0][1])

    logging.info(f"Model provided prediction {prediction}, probaibility: {probability}")

    return{
        "prediction":prediction,
        "probability":probability
    }


# example usage
# sample_input = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }
# result = predict(input_data=sample_input)
# print(result)