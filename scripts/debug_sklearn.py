import sklearn
import sys
from pathlib import Path

print(f"Python Version: {sys.version}")
print(f"Scikit-Learn Version: {sklearn.__version__}")

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

try:
    from sklearn.frozen import FrozenEstimator
    print("\nTesting CalibratedClassifierCV with FrozenEstimator...")
    cal2 = CalibratedClassifierCV(FrozenEstimator(model), cv=2) # standard cv
    cal2.fit(X, y)
    print("✓ FrozenEstimator + cv=2 test SUCCESS")
except Exception as e:
    print(f"✗ FrozenEstimator test FAILED: {e}")
