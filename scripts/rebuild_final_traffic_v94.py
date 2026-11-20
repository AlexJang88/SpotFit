"""Rebuild v94 from its processed workbook; never fall back to v51/v93."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from ml.traffic_accident_pipeline_paths import PROCESSED_V94
    if not PROCESSED_V94.is_file():
        raise FileNotFoundError(f'교통 v94 복구에는 전처리 파일이 필요합니다: {PROCESSED_V94}')
    subprocess.run([sys.executable, str(ROOT / 'ml/train_traffic_accident_v94_from_processed.py')],
                   cwd=ROOT, check=True)
    from scripts.publish_traffic_accident_v93_v94 import publish
    print(publish(94))
