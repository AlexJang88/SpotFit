from pathlib import Path

import joblib
import pandas as pd


root = Path(__file__).resolve().parents[1]
rows = []
for version in (45, 46):
    bundle = joblib.load(
        root / "models" / "교통사고" / "41차~49차" / f"{version}차" / f"교통사고_{version}차_예측모델.joblib"
    )
    coefs = bundle["model"].named_steps["model"].coef_
    total = sum(abs(coefs))
    for feature, weight in zip(bundle["features"], coefs):
        rows.append({
            "모델": f"{version}차",
            "변수": feature,
            "표준화가중치": weight,
            "절대가중치비중_pct": abs(weight) / total * 100,
            "영향방향": "증가" if weight > 0 else "감소",
        })

df = pd.DataFrame(rows)
out = root / "artifacts" / "reports" / "교통사고_45차_46차_변수가중치.xlsx"
out.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="전체가중치")
    for version in ("45차", "46차"):
        subset = df[df["모델"] == version].sort_values(
            "표준화가중치", key=lambda s: s.abs(), ascending=False
        )
        subset.to_excel(writer, index=False, sheet_name=version)

print(out)
for version in ("45차", "46차"):
    print(f"\n### {version}")
    subset = df[df["모델"] == version].sort_values(
        "표준화가중치", key=lambda s: s.abs(), ascending=False
    )
    for _, row in subset.iterrows():
        print(f"{row['변수']}\t{row['표준화가중치']:.6f}\t{row['절대가중치비중_pct']:.2f}%")
