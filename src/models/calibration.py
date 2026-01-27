import numpy as np
from sklearn.isotonic import IsotonicRegression

class SafeCalibrator:
    """
    A bug-proof replacement for CalibratedClassifierCV that works on all 
    scikit-learn versions by manually implementing isotonic calibration.
    
    This class is defined in a stable module to ensure it can be pickled
     and unpickled across different scripts.
    """
    def __init__(self, base_model, scaler=None):
        self.base_model = base_model
        self.scaler = scaler
        self.calibrators = []
        
    def fit(self, X_val, y_val):
        # Apply scaler if present
        X_val_processed = self.scaler.transform(X_val) if self.scaler else X_val
        
        # Get raw probabilities (N x 3 or N x 2)
        probs = self.base_model.predict_proba(X_val_processed)
        n_classes = probs.shape[1]
        self.calibrators = []
        
        for i in range(n_classes):
            iso = IsotonicRegression(out_of_bounds='clip')
            # target is 1 if class matches, 0 otherwise
            target = (y_val == i).astype(float)
            iso.fit(probs[:, i], target)
            self.calibrators.append(iso)
        return self
        
    def predict_proba(self, X):
        # Apply scaler if present
        X_processed = self.scaler.transform(X) if self.scaler else X
        
        probs = self.base_model.predict_proba(X_processed)
        calibrated_probs = np.zeros_like(probs)
        
        for i, iso in enumerate(self.calibrators):
            calibrated_probs[:, i] = iso.transform(probs[:, i])
            
        # Renormalize to sum to 1
        sums = calibrated_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        sums[sums == 0] = 1.0
        return calibrated_probs / sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
