from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
V52=ROOT/"models"/"교통사고"/"52차"
V53=ROOT/"models"/"교통사고"/"53차"
OUT=ROOT/"models"/"교통사고"/"54차"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model2=joblib.load(V52/"교통사고_52차_2027년_2년후_모델.joblib")
    model3=joblib.load(V53/"교통사고_53차_2028년_3년후_모델.joblib")
    model2.update(version=54,selection="52차의 2년 후 최우수 모델")
    model3.update(version=54,selection="53차의 3년 후 최우수 모델")
    joblib.dump(model2,OUT/"교통사고_54차_2027년_2년후_모델.joblib")
    joblib.dump(model3,OUT/"교통사고_54차_2028년_3년후_모델.joblib")

    d52=pd.read_excel(V52/"52차_검증결과.xlsx",sheet_name="연도별검증") if (V52/"52차_검증결과.xlsx").is_file() else None
    # 52차 technique report contains the accepted fold details.
    techniques=pd.read_excel(ROOT/"artifacts"/"reports"/"traffic_v52_techniques"/"기법전체탐색.xlsx",sheet_name="연도별검증")
    winner52=pd.read_excel(ROOT/"artifacts"/"reports"/"traffic_v52_techniques"/"기법전체탐색.xlsx",sheet_name="최종후보")
    name2=winner52.loc[winner52.예측간격.eq(2),"모델"].iloc[0]
    detail2=techniques[(techniques.예측간격==2)&(techniques.모델==name2)&(techniques.방식=="rank")]
    detail53=pd.read_excel(V53/"53차_검증결과.xlsx",sheet_name="연도별검증")
    detail3=detail53[detail53.예측간격==3]
    detail=pd.concat([detail2[["예측간격","검증목표연도","정확등급정확도","±1등급정확도","평균등급오차"]],detail3],ignore_index=True)
    summary=detail.groupby("예측간격",as_index=False).agg(exact=("정확등급정확도","mean"),within1=("±1등급정확도","mean"),error=("평균등급오차","mean"),min_exact=("정확등급정확도","min")).rename(columns={"exact":"정확등급정확도","within1":"±1등급정확도","error":"평균등급오차","min_exact":"최저연도정확도"})
    metrics={"2년후":summary.iloc[0].to_dict(),"3년후":summary.iloc[1].to_dict(),"평균정확등급":float(summary.정확등급정확도.mean()),"평균_1칸이내":float(summary["±1등급정확도"].mean()),"평균등급오차":float(summary.평균등급오차.mean())}
    info={"모델명":"교통사고 54차 장기등급 모델","기반":"52차·53차","선정방식":"예측기간별 최우수 모델 결합","2년모델":"52차 유지","3년모델":"53차 후진제거 모델","성능":metrics}
    (OUT/"model_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    with pd.ExcelWriter(OUT/"54차_검증결과.xlsx",engine="openpyxl") as w: detail.to_excel(w,sheet_name="연도별검증",index=False); summary.to_excel(w,sheet_name="요약",index=False)
    print(json.dumps(info,ensure_ascii=False,indent=2,default=float))


if __name__=="__main__": main()
