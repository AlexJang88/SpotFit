"""Restore only the three delivered final property bundles."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

if __name__ == '__main__':
    from app.services.property_model_bootstrap import ensure_models
    results = ensure_models(ROOT)
    print(results)
    raise SystemExit(1 if any(value.startswith('failed') for value in results.values()) else 0)
