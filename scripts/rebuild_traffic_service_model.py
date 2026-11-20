"""Reproduce the locked second traffic model and all API-facing artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/interim/안전/교통사고"
FALLBACK = ROOT / "models/recommender/v1/traffic_accident/교통사고_2차모델_입력데이터"
OUT = ROOT / "models/recommender/v1/traffic_accident"
FEATURES = ["인구만명당 교통사고 사망자수", "인구만명당 의료보장 사업장수",
            "운전자안전벨트 착용률", "음주운전 경험률", "현재연도 교통사고 건수"]
TARGET = "다음연도 교통사고 건수"


def source(name: str) -> Path:
    path = DATA / name
    return path if path.is_file() else FALLBACK / name


def metric(actual, prediction) -> dict:
    actual, prediction = np.asarray(actual, float), np.maximum(np.asarray(prediction, float), 0)
    error = np.abs(actual - prediction)
    wmape = error.sum() / actual.sum() * 100
    return {"MAE": float(error.mean()), "RMSE": float(np.sqrt(np.mean((actual-prediction)**2))),
            "R2": float(1 - np.sum((actual-prediction)**2) / np.sum((actual-actual.mean())**2)),
            "WMAPE(%)": float(wmape), "오차기반정확도(%)": float(max(0, 100-wmape))}


def load() -> pd.DataFrame:
    accidents = pd.read_excel(source("자치구별_교통사고_2차.xlsx"), sheet_name=0)
    drivers = pd.read_csv(source("자치구별_운전자안전_2차.csv"))
    workplaces = pd.read_csv(source("자치구별_인구만명당_의료보장사업장수_2017_2024.csv"))
    population = pd.read_csv(source("자치구별_주민등록인구_2017_2025.csv"))
    keys = ["연도", "자치구"]
    for frame in (accidents, drivers, workplaces, population):
        frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)
    base = accidents[accidents.연도.between(2017, 2025)][keys + ["사고건수", "사망자수"]].copy()
    base = base.merge(population[keys + ["주민등록인구_계"]], on=keys, validate="one_to_one")
    base = base.merge(workplaces[keys + ["인구만명당_의료보장사업장수"]], on=keys, how="left", validate="one_to_one")
    base = base.merge(drivers[keys + ["운전자안전벨트착용률", "음주운전율"]], on=keys, how="left", validate="one_to_one")
    base["인구만명당 교통사고 사망자수"] = base.사망자수 / base.주민등록인구_계 * 10_000
    base = base.rename(columns={"인구만명당_의료보장사업장수": "인구만명당 의료보장 사업장수",
                                "운전자안전벨트착용률": "운전자안전벨트 착용률",
                                "음주운전율": "음주운전 경험률", "사고건수": "현재연도 교통사고 건수"})
    base = base.sort_values(["자치구", "연도"])
    base[TARGET] = base.groupby("자치구")["현재연도 교통사고 건수"].shift(-1)
    base["예측대상연도"] = base.연도 + 1
    return base


def add_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    weights = {FEATURES[0]: 62.5, FEATURES[1]: 12.5, FEATURES[2]: 8.99/(8.99+8.70)*25,
               FEATURES[3]: 8.70/(8.99+8.70)*25}
    safer = {FEATURES[0]: False, FEATURES[1]: False, FEATURES[2]: True, FEATURES[3]: False}
    cols=[]
    for feature, weight in weights.items():
        z=out.groupby("연도")[feature].transform(lambda s:(s-s.min())/(s.max()-s.min()) if s.max()>s.min() else .5)
        col=f"점수기여_{feature}"; out[col]=(z if safer[feature] else 1-z)*weight; cols.append(col)
    out["안전점수_100점"]=out[cols].sum(axis=1)
    return out


def main() -> None:
    frame = add_score(load())
    labeled = frame[frame[TARGET].notna() & frame[FEATURES].notna().all(axis=1)].copy()
    train, test = labeled[labeled.예측대상연도 <= 2024], labeled[labeled.예측대상연도 == 2025]
    if len(train) != 175 or len(test) != 25:
        raise ValueError(f"교통 모델 시간분할 불일치: train={len(train)}, test={len(test)}")
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("model", Ridge(alpha=10.0))])
    model.fit(train[FEATURES], train[TARGET]); prediction=np.maximum(model.predict(test[FEATURES]),0)
    validation=test[["연도","예측대상연도","자치구",TARGET,"안전점수_100점"]].copy()
    validation["예측 교통사고 건수"]=np.round(prediction,1)
    metrics=metric(test[TARGET],prediction)
    importance=permutation_importance(model,test[FEATURES],test[TARGET],scoring="neg_mean_absolute_error",n_repeats=20,random_state=42)
    OUT.mkdir(parents=True,exist_ok=True)
    joblib.dump({"pipeline":model,"features":FEATURES,"target":TARGET,"selected_model":"Ridge"},OUT/"교통사고_2차_모델.joblib")
    validation.to_csv(OUT/"2025년_최종검증_예측결과.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"변수":FEATURES,"순열중요도_MAE증가":importance.importances_mean}).to_csv(OUT/"변수중요도.csv",index=False,encoding="utf-8-sig")
    metadata={"모델구분":"안전-교통사고-2차","예측목표":TARGET,"선정모델":"Ridge","입력변수":FEATURES,
              "학습행":len(train),"최종검증행":len(test),"최종검증지표":metrics}
    (OUT/"모델정보.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False))


if __name__ == "__main__": main()
