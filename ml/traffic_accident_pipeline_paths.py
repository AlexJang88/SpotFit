"""Canonical paths for the traffic-accident v93/v94 rebuild pipeline."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_V93 = ROOT / "data/raw/93차 모델 데이터"
RAW_V94 = ROOT / "data/raw/94차 모델 데이터"
RAW_V94_TRAFFIC = RAW_V94 / "교통량"
INTERIM_ROOT = ROOT / "data/interim/교통사고"
INTERIM_V94_TRAFFIC = INTERIM_ROOT / "94차/자치구별_교통량_2017_2025.csv"
PROCESSED_V93 = ROOT / "data/processed/교통사고_93차_데이터.xlsx"
PROCESSED_V94 = ROOT / "data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx"
MODEL_V93_REBUILT = ROOT / "models/교통사고/93차.Processed단독재현검증"
MODEL_V94_REBUILT = ROOT / "models/교통사고/94차.Processed단독재현"
RECOMMENDER_V1 = ROOT / "models/recommender/v1"
