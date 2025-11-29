# import os
# import numpy as np
# import pandas as pd
# import pickle
# import logging
# from sklearn.ensemble import RandomForestClassifier
# import yaml

# # Ensure the "logs" directory exists
# log_dir = 'logs'
# os.makedirs(log_dir, exist_ok=True)

# # logging configuration
# logger = logging.getLogger('model_building')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# log_file_path = os.path.join(log_dir, 'model_building.log')
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setLevel('DEBUG')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error: %s', e)
#         raise


# def load_data(file_path: str) -> pd.DataFrame:
#     """
#     Load data from a CSV file.
    
#     :param file_path: Path to the CSV file
#     :return: Loaded DataFrame
#     """
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except FileNotFoundError as e:
#         logger.error('File not found: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
#     """
#     Train the RandomForest model.
    
#     :param X_train: Training features
#     :param y_train: Training labels
#     :param params: Dictionary of hyperparameters
#     :return: Trained RandomForestClassifier
#     """
#     try:
#         if X_train.shape[0] != y_train.shape[0]:
#             raise ValueError("The number of samples in X_train and y_train must be the same.")
        
#         logger.debug('Initializing RandomForest model with parameters: %s', params)
#         clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
#         logger.debug('Model training started with %d samples', X_train.shape[0])
#         clf.fit(X_train, y_train)
#         logger.debug('Model training completed')
        
#         return clf
#     except ValueError as e:
#         logger.error('ValueError during model training: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Error during model training: %s', e)
#         raise


# def save_model(model, file_path: str) -> None:
#     """
#     Save the trained model to a file.
    
#     :param model: Trained model object
#     :param file_path: Path to save the model file
#     """
#     try:
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
#         with open(file_path, 'wb') as file:
#             pickle.dump(model, file)
#         logger.debug('Model saved to %s', file_path)
#     except FileNotFoundError as e:
#         logger.error('File path not found: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Error occurred while saving the model: %s', e)
#         raise

# def main():
#     try:
#         params = load_params('params.yaml')['model_building']
#         train_data = load_data('./data/processed/train_tfidf.csv')
#         X_train = train_data.iloc[:, :-1].values
#         y_train = train_data.iloc[:, -1].values

#         clf = train_model(X_train, y_train, params)
        
#         model_save_path = 'models/model.pkl'
#         save_model(clf, model_save_path)

#     except Exception as e:
#         logger.error('Failed to complete the model building process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()

import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml

# ---------------- logging ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------- params helper ----------------
DEFAULT_PARAMS = {
    'model_type': 'random_forest',
    'random_state': 42,
    'n_estimators': 100,
    'model_dir': './models'
}


def find_params_file(filename: str = 'params.yaml') -> Path | None:
    """Search upward from this script for params.yaml and return Path or None."""
    current = Path(__file__).resolve().parent
    for p in [current, *current.parents]:
        candidate = p / filename
        if candidate.exists():
            return candidate
    return None


def load_params_from_file(path: Path) -> dict:
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        logger.debug("Loaded params from %s", path)
        return data
    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error reading params file: %s", e)
        raise


def get_params() -> dict:
    """Return a params dict merged with defaults. No exceptions for missing file or keys."""
    params_path = find_params_file()
    if params_path is None:
        logger.warning("params.yaml not found. Using DEFAULT_PARAMS: %s", DEFAULT_PARAMS)
        return DEFAULT_PARAMS.copy()

    loaded = load_params_from_file(params_path)
    # Attempt to fetch model_building section, fall back to top-level if not found
    mb = loaded.get('model_building') or loaded.get('model') or {}
    # Merge shallowly with defaults
    merged = DEFAULT_PARAMS.copy()
    for k, v in mb.items():
        merged[k] = v
    # ensure numeric defaults exist
    merged.setdefault('n_estimators', DEFAULT_PARAMS['n_estimators'])
    merged.setdefault('random_state', DEFAULT_PARAMS['random_state'])
    merged.setdefault('model_dir', DEFAULT_PARAMS['model_dir'])
    return merged


# ---------------- data / model helpers ----------------
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s", file_path, df.shape)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Error loading data %s: %s", file_path, e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> RandomForestClassifier:
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train have different number of rows")

    n_estimators = int(cfg.get('n_estimators', DEFAULT_PARAMS['n_estimators']))
    random_state = int(cfg.get('random_state', DEFAULT_PARAMS['random_state']))

    logger.debug("Training RandomForest (n_estimators=%s, random_state=%s)", n_estimators, random_state)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    logger.debug("Model training done")
    return clf


def save_model(model, output_dir: str, filename: str = 'model.pkl'):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full = out_path / filename
    with open(full, 'wb') as f:
        pickle.dump(model, f)
    logger.debug("Model saved to %s", full)
    return full


# ---------------- main ----------------
def main():
    try:
        cfg = get_params()
        logger.debug("Using params: %s", cfg)

        # load processed tfidf data
        train_fp = Path('./data/processed/train_tfidf.csv')
        if not train_fp.exists():
            logger.error("Train file not found: %s", train_fp)
            raise FileNotFoundError(f"{train_fp} does not exist. Run feature_engineering first.")

        train_df = load_data(str(train_fp))
        # expect last column to be label
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        model = train_model(X_train, y_train, cfg)
        model_file = save_model(model, cfg.get('model_dir', DEFAULT_PARAMS['model_dir']))

        logger.info("Model training pipeline completed. Saved: %s", model_file)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
