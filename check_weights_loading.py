
import sys
from pathlib import Path
import joblib

# Add src to path
sys.path.append(str(Path.cwd()))

def check_model_type():
    model_path = Path('models/catboost_v1_latest.pkl')
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        print(f"Type: {type(model)}")
        print(f"Class Name: {model.__class__.__name__}")
        
        if hasattr(model, 'get_param'):
            print(f"Loss function: {model.get_param('loss_function')}")
            
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    check_model_type()
