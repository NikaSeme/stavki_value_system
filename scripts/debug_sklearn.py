from sklearn.isotonic import IsotonicRegression
import numpy as np

try:
    from sklearn.frozen import FrozenEstimator
    print("✓ FrozenEstimator import SUCCESS (from sklearn.frozen)")
except ImportError:
    try:
        from sklearn.utils import FrozenEstimator
        print("✓ FrozenEstimator import SUCCESS (from sklearn.utils)")
    except ImportError:
        print("✗ FrozenEstimator import FAILED")

from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import numpy as np

print("\nTesting CalibratedClassifierCV with cv='prefit'...")
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, 10)
model = CatBoostClassifier(iterations=1, verbose=False).fit(X, y)

try:
    print("\nTesting CalibratedClassifierCV with cv='prefit' AND ensemble=False...")
    cal3 = CalibratedClassifierCV(model, cv='prefit', ensemble=False)
    cal3.fit(X, y)
    print("✓ cv='prefit' + ensemble=False test SUCCESS")
except Exception as e:
    print(f"✗ cv='prefit' + ensemble=False test FAILED: {e}")

try:
    print("\nTesting CalibratedClassifierCV with cv='prefit' (standard)...")
    cal = CalibratedClassifierCV(model, cv='prefit')
    cal.fit(X, y)
    print("✓ cv='prefit' test SUCCESS")
except Exception as e:
    print(f"✗ cv='prefit' test FAILED: {e}")

class SafeCalibrator:
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrators = []
    def fit(self, X_val, y_val):
        probs = self.base_model.predict_proba(X_val)
        n_classes = probs.shape[1]
        for i in range(n_classes):
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs[:, i], (y_val == i).astype(float))
            self.calibrators.append(iso)
        return self
    def predict_proba(self, X):
        probs = self.base_model.predict_proba(X)
        calibrated = np.zeros_like(probs)
        for i, iso in enumerate(self.calibrators):
            calibrated[:, i] = iso.transform(probs[:, i])
        sums = calibrated.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return calibrated / sums

print("\nTesting SafeCalibrator (Bug-Proof Manual Implementation)...")
try:
    sc = SafeCalibrator(model)
    sc.fit(X, y)
    probs_cal = sc.predict_proba(X)
    print(f"✓ SafeCalibrator SUCCESS. Output shape: {probs_cal.shape}")
    print(f"✓ Probabilities sum to 1: {np.allclose(probs_cal.sum(axis=1), 1.0)}")
except Exception as e:
    print(f"✗ SafeCalibrator FAILED: {e}")
