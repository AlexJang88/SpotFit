from pathlib import Path
from copy import deepcopy
import shutil
import json
import joblib
import numpy as np
import pandas as pd
from pptx import Presentation
from ml.train_traffic_accident_v51_horizon_models import predict

ROOT = Path(__file__).resolve().parents[1]
PPT = next((ROOT / 'docs').glob('*51*템플릿*.pptx'))
AUDIT = ROOT / 'artifacts' / 'reports' / 'v51_ppt_review'
AUDIT.mkdir(parents=True, exist_ok=True)
BACKUP = AUDIT / ('원본_' + PPT.name)
if not BACKUP.exists():
    shutil.copy2(PPT, BACKUP)
prs = Presentation(BACKUP)

def replace(frame, text):
    # Retain paragraph properties and run formatting, including template fonts.
    templates = [deepcopy(p._p) for p in frame.paragraphs]
    for p in list(frame._txBody):
        if p.tag.endswith('}p'):
            frame._txBody.remove(p)
    for i, line in enumerate(text.split('\n')):
        element = deepcopy(templates[min(i, len(templates)-1)])
        frame._txBody.append(element)
        paragraph = frame.paragraphs[-1]
        runs = list(paragraph.runs)
        if runs:
            runs[0].text = line
            for run in runs[1:]:
                paragraph._p.remove(run._r)
        else:
            paragraph.add_run().text = line

def text(slide, shape, value):
    replace(prs.slides[slide-1].shapes[shape].text_frame, value)

def cell(slide, shape, row, col, value):
    replace(prs.slides[slide-1].shapes[shape].table.cell(row,col).text_frame, value)

text(1,7,'2025년 기준 입력으로 2027·2028년의 자치구별 상대 안전등급 예측')
text(1,24,'안전 1등급 가장 안전 · 안전 5등급 가장 위험 | 안전순위별 5개 자치구 배정 | 수치는 과거 검증 결과')
text(2,4,'교통안전모델 ② · 목표연도별 직접예측')
text(2,7,'같은 2025년 기준 입력에서 2년 후 모델과 3년 후 모델을 각각 적용')
for shape in (13,25,33):
    text(2,shape,'2025 기준 입력')
text(2,41,'예측 간격은 입력연도 기준입니다. 2027·2028년의 실제 성능은 아직 확인할 수 없습니다.')
cell(3,8,2,1,'2017→2019 … 2023→2025 (175행)')
cell(3,8,2,2,'2017→2020 … 2022→2025 (150행)')
cell(3,8,3,1,'2025년 기준 전처리 데이터')
cell(3,8,3,2,'2025년 기준 전처리 데이터')
text(3,9,'학습 목표: 미래 종합위험점수 또는 현재 대비 변화량')
text(3,10,'내부 위험점수 순위를 반전해 안전순위·안전등급을 산출합니다. 안전 1등급이 가장 안전합니다.')
text(4,7,'두 구성 예측에 현재 점수를 각각 40% 혼합한 뒤, 75:25로 결합')
text(4,26,'GBR depth 3 · 입력 39개\n미래점수 60% + 현재점수 40%\n최종 반영 비중: 75%')
text(4,30,'Ridge α=0.1 · 입력 26개\n현재점수 + 예측변화량 × 60%\n최종 반영 비중: 25%')
text(4,35,'2025년 기준 입력 → 2027 예측 | 두 구성의 고유 입력 합계 39개 · 자치구 원-핫 없음')
text(5,26,'GBR depth 3 · 입력 64개\n3년 후 종합위험점수 예측\n최종 반영 비중: 50%')
text(5,30,'Ridge α=30 · 입력 64개\n3년 후 종합위험점수 예측\n최종 반영 비중: 50%')
text(5,35,'2025년 기준 입력 → 2028 예측 | 고유 입력 64개 · 자치구 원-핫 25개 포함 · 현재점수 별도 혼합 없음')
text(6,4,'교통안전모델 ⑥ · 입력 데이터와 검증 기준')
text(6,7,'기본 지표와 파생변수를 입력하고, 자치구별 2년 후·3년 후 목표를 연결')
cell(6,8,2,0,'2년 모델 입력')
cell(6,8,2,1,'기본 지표 13개 + 5분위 점수 13개 + 점수·등급·시계열 13개')
cell(6,8,3,0,'3년 모델 입력')
cell(6,8,3,1,'2년 모델의 39개 + 자치구 원-핫 25개 = 64개')
cell(6,8,4,0,'점수·등급 정의')
cell(6,8,4,1,'13개 지표의 5분위 점수에 49차 계수 기반 가중치를 적용')
cell(6,8,5,0,'기존 검증 방식')
cell(6,8,5,1,'검증 목표연도보다 이른 목표연도의 학습쌍으로 재학습')
cell(6,8,6,0,'예측 입력')
cell(6,8,6,1,'2025년 기준 25행 → 2027·2028년 각각 25개 자치구')
text(6,9,'과거 검증: 2년 모델 2021~2025년 / 3년 모델 2022~2025년')
text(6,10,'공식 기준모델은 Persistence(현재 안전등급 유지)입니다. 정확도는 2년 48.8%·3년 34.0%, 51차는 56.0%·55.0%입니다.')
text(8,4,'교통안전모델 ⑦ · 현재 서비스 제공 방식')
text(8,5,'Service · Stored Forecasts for 2027 and 2028')
text(8,7,'51차 모델과 저장된 예측 결과를 이용하는 발표용 임시 서비스')
text(8,11,'2025년 기준 · 2년 후\n안전점수 · 안전순위 · 안전 5등급\n서울 25개 자치구')
text(8,15,'2025년 기준 · 3년 후\n안전점수 · 안전순위 · 안전 5등급\n서울 25개 자치구')
text(8,18,'서버 응답 방식')
text(8,19,'저장된 예측 CSV 조회\n예측연도별 25개 결과 반환\n안전순위 기준 정렬')
cell(8,20,0,1,'현재 구현')
cell(8,20,1,0,'응답 정보')
cell(8,20,1,1,'자치구 · 기준연도 · 예측연도 · 예측간격 · 안전점수 · 안전순위 · 안전등급')
cell(8,20,2,0,'서비스 파일 누락 시')
cell(8,20,2,1,'51차 원본 모델 복사 + processed 예측 시트에서 서비스 CSV 생성')
cell(8,20,3,0,'새 데이터 반영')
cell(8,20,3,1,'학습·예측 결과 갱신 후 서비스 파일 재생성 필요')
text(8,21,'서비스 경로: models/recommender/v1/traffic_accident_v51 | 서버 초기화는 재학습이 아닌 서비스 파일 구성입니다.')
# Remove redundant performance comparison with an unrelated 1-year model/chart.
slide_id = prs.slides._sldIdLst[6]
prs.part.drop_rel(slide_id.rId)
prs.slides._sldIdLst.remove(slide_id)
prs.save(PPT)

service = ROOT / 'models/recommender/v1/traffic_accident_v51'
data = pd.read_excel(ROOT / 'data/processed/교통사고_51차_최종모델데이터.xlsx',sheet_name='51차_2년3년목표데이터')
csv = pd.read_csv(service/'service_forecast.csv',encoding='utf-8-sig')
checks = {}
for h in (2,3):
    b = joblib.load(service/f'traffic_accident_{h}y.joblib')
    future = data.loc[data['연도'].eq(2025)].copy()
    values = predict(b,future)
    joined = pd.DataFrame({'district':future['자치구'],'recomputed':values}).merge(csv.loc[csv.prediction_year.eq(2025+h)],on='district',validate='one_to_one')
    assert np.allclose(joined.recomputed,joined.predicted_risk_score)
    checks[h] = {'component_features':[len(b[k]['features']) for k in ('component_a','component_b')], 'forecast_csv_match':True, 'training_rows':int(data[f'{h}년후_종합위험점수'].notna().sum()), 'reported_metrics':b['best_model']}
assert len(Presentation(PPT).slides)==7
(AUDIT/'검수결과.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'ppt':str(PPT),'slides':7,'checks':checks},ensure_ascii=False,indent=2))
