# 전세 v33 (입력 파일 버전): AptJeonseForecastV33과 같은 모델·항목·평가인데, 데이터는 data/processed/전세_V33.csv 한 파일만 읽음
#   입력 파일은 ml/etl/BuildJeonseInput.py가 만듦 (전세 거래 + 매매 거래 + 주택담보대출 금리)
#   원본 v33과 달라진 점: ① 읽는 파일 1개 ② 항목 5개에 안 쓰는 계산·파일(위치·한강공원·업무지구·월세 비중·거래량 등) 제거
#                          ③ 날짜를 코드에서 따로 자르지 않음. 끝 달 = 전세·매매·금리가 모두 있는 마지막 달 (현재 데이터에서는 2026-09)
#                             (전세 거래는 2026-10-20까지 있지만 매매·금리가 9월까지라서 전세가율 등 항목을 만들 수 없음)
#   항목(5개): 6개월 상승률(%), 상대 가격(서울 대비), 금리 12개월 변화(%p), 전세가율(%), 신축 건물 거래 비율(%)
#   y = 전세 계약의 평당 보증금(만원) 동네 중앙값 (거래 10건 이상). 정답은 현재 가격 대비 비율(로그), 제곱 오차 학습, 기준 시점 e(매월 말): 정답 = e+h개월 시점의 직전 12개월 동네 가격 (h=12, 24, 36)
#   모든 항목은 e까지의 거래로만 계산. 평가: 시간 순. 기준선: 그대로, 서울 변화율 규칙
# 실행: python AptJeonseForecastV33Input.py   (결과 이름: 아파트전세_v33_입력_*, 원본 v33 결과는 덮어쓰지 않음)
import json, os
import joblib, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
SRC, OUT = BASE + 'processed/전세_V33.csv', BASE + 'processed/output/'

Y, M6, REL = '평당 가격(만원)', '6개월 상승률(%)', '상대 가격(서울 대비)'
RT, RTCHG, JR, NEW = '주택담보대출 금리(연%)', '금리 12개월 변화(%p)', '전세가율(%)', '신축 건물 거래 비율(%)'
FEAT = [M6, REL, RTCHG, JR, NEW]
STARTS = {12: range(2015, 2026), 24: range(2015, 2025), 36: range(2016, 2024)}
DEV_LAST = {12: 2021, 24: 2021, 36: 2020}                      # 항목 고를 때는 이 해까지의 시험만 사용, 그 뒤 시험은 '확인용'으로 따로 봄
NAMES = {12: '1년', 24: '2년', 36: '3년'}
RATIO = lambda df: np.log(df['정답'] / df['현재 가격'])

def xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, n_jobs=1, random_state=0)

def score(true, hat):
    err = (hat - true).abs() / true * 100
    r2 = 1 - ((true - hat) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    return err.median(), (err <= 10).mean() * 100, (hat - true).abs().mean(), r2

class Win:                                                      # 월 번호로 정렬된 표에서 기간 조각을 빠르게 자름
    def __init__(self, df): self.df = df.sort_values('m').reset_index(drop=True); self.m = self.df['m'].values
    def get(self, e, n=12):
        return self.df.iloc[np.searchsorted(self.m, e - n + 1, 'left'):np.searchsorted(self.m, e, 'right')]

class Lab:
    def __init__(self):
        x = pd.read_csv(SRC, low_memory=False, encoding='utf-8-sig')
        x['동'] = x['행정동_코드(상권분석)'].astype(int)
        dt = pd.to_datetime(x['계약일']); x['m'] = dt.dt.year * 12 + dt.dt.month - 1
        x['신축'] = (x['건축년도'] >= dt.dt.year - 5).astype(float)                       # 건축 5년 이내 (건축년도가 비면 0)
        j, s = x[x['구분'] == '전세'].copy(), x[x['구분'] == '매매']
        self.RATE = x.dropna(subset=[RT]).groupby('m')[RT].first().to_dict()           # 월 번호 → 금리
        self.m_lo = int(j['m'].min())
        self.m_hi = int(min(j['m'].max(), s['m'].max(), max(self.RATE)))              # 전세·매매·금리가 모두 있는 마지막 달
        self.a = Win(j)                                                                # 전세 거래
        self.wj = Win(j[['동', 'm', Y]].rename(columns={Y: 'y'}))
        self.ws = Win(s.dropna(subset=['m', '동', Y])[['동', 'm', Y]])                  # 매매 거래 (전세가율 분모)
        self.d = j[j['m'] <= self.m_hi].groupby('동')[['자치구', '행정동']].last().rename_axis('행정동_코드')
        self.ends = list(range(self.m_lo + 11, self.m_hi + 1))
        self._prepare()
        self.D = {h: self.build(h) for h in STARTS}

    def price(self, w, key='동', min_n=10):
        g = w.groupby(key)[Y].agg(['median', 'size']); return g['median'].where(g['size'] >= min_n)

    def seoul(self, w):
        return w[Y].median()

    def _prepare(self):
        d, idx = self.d, self.d.index
        P6, P, S, PJ, PS = {}, {}, {}, {}, {}
        for e in range(self.m_lo + 5, self.m_hi + 1): P6[e] = self.price(self.a.get(e, 6)).reindex(idx)
        for e in self.ends:
            w = self.a.get(e)
            P[e], S[e] = self.price(w).reindex(idx), self.seoul(w)
            wj = self.wj.get(e); g = wj.groupby('동')['y'].agg(['median', 'size']); PJ[e] = g['median'].where(g['size'] >= 10).reindex(idx)
            ws = self.ws.get(e); g = ws.groupby('동')[Y].agg(['median', 'size']); PS[e] = g['median'].where(g['size'] >= 10).reindex(idx)
        self.P, self.S, self.P6 = P, S, P6
        F = {}
        for e in self.ends:
            w = self.a.get(e)
            f = pd.DataFrame(index=idx)
            f[NEW] = w.groupby('동')['신축'].mean().reindex(idx) * 100
            f[M6] = (P6[e] / P6[e - 6] - 1) * 100 if (e - 6) in P6 else np.nan
            f[REL] = P[e] / S[e]
            f[RTCHG] = self.RATE.get(e, np.nan) - self.RATE.get(e - 12, np.nan)
            f[JR] = PJ[e] / PS[e] * 100
            F[e] = f
        self.F = F

    def feats(self, e):
        df = self.F[e].copy()
        df['현재 가격'], df['e'] = self.P[e], e
        return df

    def build(self, h):
        rows = []
        for e in self.ends:
            if (e + h) in self.P:
                df = self.feats(e); df['정답'] = self.P[e + h]; rows.append(df)
        return pd.concat(rows).dropna(subset=['현재 가격', '정답'])

    def tests(self, feat, h, last_year=None):
        D = self.D[h]; rows = []; last = None
        for ym in STARTS[h]:
            if last_year and ym > last_year: continue
            lo = ym * 12; hi = lo + 11
            tr = D[D['e'] + h <= lo]; te = D[(D['e'] >= lo) & (D['e'] <= hi)]
            if len(te) == 0 or tr['e'].nunique() < 24: continue
            model = xgb().fit(tr[feat], RATIO(tr)); last = (ym, model, te)
            sg = te['e'].map(lambda e: self.S[e] / self.S[e - 12] - 1 if (e - 12) in self.S else 0.0)
            preds = {'그대로': te['현재 가격'], '서울 변화율 규칙': te['현재 가격'] * (1 + sg) ** (h // 12), '예측 모델': te['현재 가격'] * np.exp(model.predict(te[feat]))}
            for meth, hat in preds.items():
                mid, w10, mae, r2 = score(te['정답'], hat)
                rows.append({'기간': NAMES[h], '시험연도': ym, '시험': f'{ym}년 기준', '방법': meth, '오차율 중앙(%)': mid, '10% 이내(%)': w10, '평균 오차(만원/평)': mae, '결정계수 R²': r2})
        return pd.DataFrame(rows), last

    def report(self, feat, tag, label):
        pre = '아파트전세'
        print(f'===== {label} {tag} | 전세 평당 보증금 | 항목 {len(feat)}개: {feat}')
        order = ['그대로', '서울 변화율 규칙', '예측 모델']; col = ['오차율 중앙(%)', '10% 이내(%)', '평균 오차(만원/평)', '결정계수 R²']
        allres = []
        for h in STARTS:
            nm = NAMES[h]; res, (ym, model, te) = self.tests(feat, h); allres.append(res)
            avg = res.groupby('방법')[col].mean().round(3).reindex(order)
            print(); print(f'[{nm} 뒤: 시험 {res["시험"].nunique()}번 평균]'); print(avg.to_string())
            hold = res[res['시험연도'] > DEV_LAST[h]].groupby('방법')[['오차율 중앙(%)', '결정계수 R²']].mean().round(3).reindex(order)
            print(f'  (항목 고를 때 쓴 구간 {int(res[res["시험연도"] <= DEV_LAST[h]]["시험연도"].nunique())}번 / 확인용 구간 {int(res[res["시험연도"] > DEV_LAST[h]]["시험연도"].nunique())}번)')
            print('  확인용 구간:', ' | '.join(f'{k} {v["오차율 중앙(%)"]}%/{v["결정계수 R²"]}' for k, v in hold.iterrows()))
            pi = permutation_importance(model, te[feat], RATIO(te), n_repeats=5, random_state=0)
            imp = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)
            print(f'[{nm} 뒤 항목 중요도 ({ym}년 시험 구간)]'); print(imp.head(8).round(3).to_string())
        pd.concat(allres).drop(columns=['시험연도']).to_csv(OUT + f'{pre}_{tag}_시험별결과.csv', index=False, encoding='utf-8-sig')
        now = self.feats(self.m_hi)
        cur = '현재 평당 보증금(최근12개월 실거래, 만원)'
        out = pd.DataFrame({'자치구': self.d['자치구'], '행정동': self.d['행정동'], cur: now['현재 가격'].round(0)})
        print(); print(f'===== {tag} 1년·2년·3년 뒤 예측 ({self.m_hi // 12}-{self.m_hi % 12 + 1:02d} 기준) =====')
        for h in STARTS:
            nm = NAMES[h]; D = self.D[h]
            model = xgb().fit(D[feat], RATIO(D))
            pred = (now['현재 가격'] * np.exp(model.predict(now[feat]))).where(now['현재 가격'].notna())
            out[f'{nm} 뒤 예측(만원/평)'] = pred.round(0)
            out[f'{nm} 뒤 예상 변화율(%)'] = ((pred / now['현재 가격'] - 1) * 100).round(1)
            ch = out[f'{nm} 뒤 예상 변화율(%)'].dropna()
            fit_r2 = score(D['정답'], D['현재 가격'] * np.exp(model.predict(D[feat])))[3]
            print(f'[{nm} 뒤] 학습 시점 {D["e"].nunique()}개 | 변화율 중앙 {ch.median():+.1f}% (범위 {ch.min():+.0f}%~{ch.max():+.0f}%, ±30% 넘는 동네 {int((ch.abs() > 30).sum())}곳) | 학습 R² {fit_r2:.3f}')
            joblib.dump(model, OUT + f'{pre}_{tag}_{nm}뒤모델.joblib')
        out['비고'] = np.where(now['현재 가격'].notna(), '', '최근12개월 거래 부족')
        out.reset_index().to_csv(OUT + f'{pre}_{tag}_동네별예측_1년2년3년뒤.csv', index=False, encoding='utf-8-sig')
        short = out.reset_index()[['행정동_코드', '자치구', '행정동', cur, '1년 뒤 예측(만원/평)', '2년 뒤 예측(만원/평)', '3년 뒤 예측(만원/평)']].dropna(subset=['1년 뒤 예측(만원/평)'])
        with open(OUT + f'{pre}_{tag}_동네별예측_1년2년3년뒤.json', 'w', encoding='utf-8') as f:
            json.dump(json.loads(short.to_json(orient='records', force_ascii=False)), f, ensure_ascii=False, indent=2)
        print(); print(f'저장: {pre}_{tag}_동네별예측_1년2년3년뒤.json ({len(short)}곳)')

if __name__ == '__main__':
    Lab().report(FEAT, 'v33_입력', '전세')
