from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = BASE_DIR / "models" / "attrition_pipeline.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"

