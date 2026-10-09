import os
import logging
from pathlib import Path


import pandas as pd
import numpy as np
from joblib import dump
from dotenv import load_dotenv


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import(
    accuracy_score, 
    classification_report,
    recall_score,
    f1_score
)

def train_model():
    try:
        #load env file content 
        load_dotenv()
        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIRS") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        # load data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape {df.shape}")

        # Seperate X and Y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        row_signature = pd.util.hash_pandas_object(X, index=False)

        # Group-based split
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups=row_signature))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Training the model
        model = Pipeline(
            steps=[
                ("scaler",StandardScaler()),
                ("model",RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    bootstrap=True,
                    ccp_alpha=0.0017,
                    max_depth=5,
                    max_features="sqrt",
                    max_samples=0.6,
                    min_samples_leaf=11,
                    min_samples_split=30,
                    n_estimators=1119
                ))
            ]
        )
        model.fit(X_train, y_train)

        # Evaluate using metrics
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train,y_train_pred)
        train_recall = accuracy_score(y_train, y_train_pred)
        train_f1 = accuracy_score(y_train, y_train_pred)

        test_acc = accuracy_score(y_test,y_test_pred)
        test_recall = accuracy_score(y_test, y_test_pred)
        test_f1 = accuracy_score(y_test, y_test_pred)

        logging.info(f"Train accuracy: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        logging.info(f"Test accuracy: {test_acc:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

        logging.info("Train classification report:\n" + classification_report(y_train, y_train_pred))
        logging.info("Test classification report:\n" + classification_report(y_test, y_test_pred))

        model.fit(X_train, y_train)
        logging.info("Model training completed")


        # save trained model
        dump(model, MODEL_PATH)
        logging.info(f"Model saved to: {MODEL_PATH}")

        logging.info("training script completed")

    except Exception as e:
        print(f"Training failed: {e}") 
        logging.exception(f"Training Script Failed: {e}") 
        raise


if __name__ =="__main__":
    train_model()      




