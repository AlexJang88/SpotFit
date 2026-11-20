# 월세 v15 아파트 월세 1년·2년·3년 뒤 평당 월세 예측 (2011~2026 합본 전월세 계약, XGBoost, 시간 순 평가) + 동네별 JSON
#   y = 월세 계약의 평당 월세(만원) 동네 구성 고정 가격 (월세 v8과 같음: 보증금÷(월세×12) 전체 중앙값 이하/초과 두 그룹의 중앙값을 반반 평균, 두 그룹 모두 있고 합 10건 이상). 학습: 2011년 이후 전체 기준 시점, 제곱 오차, 정답은 현재 가격 대비 비율(로그)
#   가설: 신규 공급(주택인허가 기준 자치구)이 월세를 누른다: 착공 후 미준공 세대수, 최근3년 허가 세대수, 최근1년 준공 세대수 추가
#   항목(6개): 6개월 상승률(%), 12개월 상승률(%), 상대 가격(서울 대비), 자치구 착공 후 미준공 세대수, 자치구 최근3년 허가 세대수, 자치구 최근1년 준공 세대수
#   기준 시점 e(매월 말): 정답 = e+h개월 시점의 직전 12개월 동네 가격 (h=12: 1년, 24: 2년, 36: 3년). 모든 항목은 e까지의 자료로만 계산
#   평가: 시간 순. 기준선: 그대로, 서울 변화율 규칙. 시험 연도를 '앞 구간'(1·2년 2021년까지, 3년 2020년까지)과 '뒤 구간'으로 나눠 둘 다 출력
# 실행: python AptWolseForecastV15.py   (내 PC에서만, 서버 기록 없음)
import json
import joblib, numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

import os
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
DATA, RAW, OUT = BASE + 'processed/', BASE + 'raw/', BASE + 'processed/output/'

M6, M12, REL = '6개월 상승률(%)', '12개월 상승률(%)', '상대 가격(서울 대비)'
G24, SEOUL12, GU12 = '24개월 상승률(%)', '서울 12개월 상승률(%)', '자치구 12개월 상승률(%)'
VOL, VOLCHG, CV = '거래량(최근12개월, 로그)', '거래량 변화(%)', '가격 변동계수(%)'
FLOOR, AGE, AREA, NEW = '층수', '완공연도', '평균 임대면적(㎡)', '신축 건물 거래 비율(%)'
HG = '가장 가까운 한강공원까지 거리(km)'
BIZC = ['강남(GBD)까지 거리(km)', '광화문(CBD)까지 거리(km)', '여의도(YBD)까지 거리(km)']
RT, RTCHG = '주택담보대출 금리(연%)', '금리 12개월 변화(%p)'
SALE12, JR, WSH, J12 = '매매 12개월 상승률(%)', '전세가율(%)', '월세 계약 비중(%)', '전세 12개월 상승률(%)'
GUREL, ACC, PEAK = '자치구 대비 상대 가격', '가격 가속도(6개월×2−12개월)', '최근 36개월 고점 대비(%)'
SEOUL6, SEOUL24, SALE6, JRCHG, NB12, W12 = '서울 6개월 상승률(%)', '서울 24개월 상승률(%)', '매매 6개월 상승률(%)', '전세가율 12개월 변화(%p)', '주변 5개 동 12개월 상승률(%)', '월세 12개월 상승률(%)'
GAPP, GSTART, GPIPE, GDONE, DDONE = '자치구 최근3년 허가 세대수', '자치구 최근3년 착공 세대수', '자치구 착공 후 미준공 세대수', '자치구 최근1년 준공 세대수', '동 최근3년 준공 세대수(K-apt)'
BIZ = {'강남(GBD)': (127.0276, 37.4979), '광화문(CBD)': (126.9770, 37.5657), '여의도(YBD)': (126.9242, 37.5216)}
STARTS = {12: range(2015, 2026), 24: range(2015, 2025), 36: range(2016, 2024)}
DEV_LAST = {12: 2021, 24: 2021, 36: 2020}                      # 항목 고를 때는 이 해까지의 시험만 사용, 그 뒤 시험은 '확인용'으로 따로 봄
NAMES = {12: '1년', 24: '2년', 36: '3년'}
RATIO = lambda df: np.log(df['정답'] / df['현재 가격'])

def xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, n_jobs=1, random_state=0)

def haversine_km(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))

def score(true, hat):
    err = (hat - true).abs() / true * 100
    r2 = 1 - ((true - hat) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    return err.median(), (err <= 10).mean() * 100, (hat - true).abs().mean(), r2

class Win:                                                      # 월 번호로 정렬된 표에서 기간 조각을 빠르게 자름
    def __init__(self, df): self.df = df.sort_values('m').reset_index(drop=True); self.m = self.df['m'].values
    def get(self, e, n=12):
        return self.df.iloc[np.searchsorted(self.m, e - n + 1, 'left'):np.searchsorted(self.m, e, 'right')]

class Lab:
    def __init__(self, kind):
        self.kind = kind
        self.Y = '평당 월세(만원)' if kind == 'wolse' else '평당 보증금(만원)'
        cols = ['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '전월세구분', '보증금(만원)', '월세(만원)', '평당 보증금(만원)', '평당 월세(만원)', '층', '건축년도', '임대면적(㎡)']
        x = pd.read_csv(DATA + '../raw/부동산/아파트_전월세_2010_2026.csv', low_memory=False, usecols=cols)
        x['dt'] = pd.to_datetime(x['계약일'], errors='coerce')
        x = x[x['dt'].notna() & (x['dt'] <= pd.Timestamp('2026-09-30'))].copy()
        x['동'] = x['행정동_코드(상권분석)'].astype(int)
        x['m'] = x['dt'].dt.year * 12 + x['dt'].dt.month - 1
        x['월세계약'] = (x['전월세구분'] == '월세').astype(float)
        self.wall = Win(x[['동', 'm', '월세계약']])
        if kind == 'jeonse': a = x[x['전월세구분'] == '전세'].copy()
        else:
            a = x[(x['전월세구분'] == '월세') & (x['월세(만원)'] > 0)].copy()
            a['배수'] = a['보증금(만원)'] / (a['월세(만원)'] * 12)
            a['bin'] = (a['배수'] > a['배수'].quantile(0.5)).astype(int)         # 0 = 보증금 작은 쪽, 1 = 큰 쪽
        a['신축'] = (a['건축년도'] >= a['dt'].dt.year - 5).astype(float)
        self.a = Win(a)
        self.wj = Win(x[(x['전월세구분'] == '전세')][['동', 'm', '평당 보증금(만원)']].rename(columns={'평당 보증금(만원)': 'y'}))
        s = pd.read_csv(DATA + '../raw/부동산/아파트_매매_2010_2026.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '계약일', '평당 금액(만원)'])
        s['m'] = pd.to_datetime(s['계약일'], errors='coerce').dt.year * 12 + pd.to_datetime(s['계약일'], errors='coerce').dt.month - 1
        s['동'] = s['행정동_코드(상권분석)']
        self.ws = Win(s.dropna(subset=['m', '동', '평당 금액(만원)']).astype({'동': int, 'm': int}))
        pm = pd.read_csv(RAW + '부동산/서울시 주택인허가 기본개요.csv', encoding='cp949', low_memory=False)
        pm = pm[pm['총_세대_수'] >= 30].copy()
        pm['구'] = pm['시군구코드명'].astype(str).str.split().str[-1].str.strip()
        for c_, n_ in [('승인_일', 'ma'), ('실제착공일', 'ms'), ('사용검사일자', 'mc')]:
            t_ = pd.to_datetime(pm[c_], errors='coerce'); pm[n_] = t_.dt.year * 12 + t_.dt.month - 1
        self.pm = pm[['구', '총_세대_수', 'ma', 'ms', 'mc']]
        kp = pd.read_csv(BASE + 'raw/부동산/단지_행정동.csv', encoding='utf-8-sig', low_memory=False)
        t_ = pd.to_datetime(kp['사용승인일'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        kp['m'] = t_.dt.year * 12 + t_.dt.month - 1; kp['세대'] = pd.to_numeric(kp['세대수'], errors='coerce')
        self.kp = kp.dropna(subset=['m', '세대'])[['행정동_코드', 'm', '세대']]
        wr = x[(x['전월세구분'] == '월세') & (x['월세(만원)'] > 0)][['동', 'm', '평당 월세(만원)']].rename(columns={'평당 월세(만원)': 'y'})
        self.wr = Win(wr)
        d = a.groupby('동')[['자치구', '행정동']].last().rename_axis('행정동_코드')
        self.d = d
        area = pd.read_csv(RAW + '부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드').reindex(d.index)
        hp = pd.read_csv(RAW + '부동산/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
        hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)
        st = pd.DataFrame(index=d.index)
        st[HG] = cKDTree(np.c_[hx, hy]).query(area[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000
        lon, lat = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True).transform(area['엑스좌표_값'].values, area['와이좌표_값'].values)
        for k, (lo, la) in BIZ.items(): st[f'{k}까지 거리(km)'] = haversine_km(lo, la, lon, lat)
        self.static = st
        xy = np.c_[area['엑스좌표_값'].values, area['와이좌표_값'].values]
        self.nb = np.argsort(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1), axis=1)[:, 1:6]
        rt = pd.read_csv(RAW + '부동산/예금은행_대출금리_신규취급액_기준__20261106130754.csv', encoding='cp949')
        rt = pd.to_numeric(rt[rt['계정항목별'].str.strip() == '주택담보대출 (연리%)'].iloc[0, 1:], errors='coerce')
        self.RATE = {int(c[:4]) * 12 + int(c[5:7]) - 1: v for c, v in rt.items()}
        self.m_lo, self.m_hi = int(a['m'].min()), int(a['m'].max())
        self.ends = list(range(self.m_lo + 11, self.m_hi + 1))
        self._prepare()
        self.D = {h: self.build(h) for h in STARTS}

    def price(self, w, key='동', min_n=10):
        Y = self.Y
        if self.kind == 'jeonse':
            g = w.groupby(key)[Y].agg(['median', 'size']); return g['median'].where(g['size'] >= min_n)
        gg = w.groupby([key, 'bin'])[Y].agg(['median', 'size']).unstack('bin')
        sz = gg['size'].reindex(columns=[0, 1])
        ok = (sz >= 1).all(axis=1) & (sz.sum(axis=1) >= min_n)
        return gg['median'].reindex(columns=[0, 1]).mean(axis=1).where(ok)

    def seoul(self, w):
        return w[self.Y].median() if self.kind == 'jeonse' else w.groupby('bin')[self.Y].median().mean()

    def _prepare(self):
        d, idx = self.d, self.d.index
        P6, P, S, PG, CNT = {}, {}, {}, {}, {}
        PJ, PS, PS6, S6, PW = {}, {}, {}, {}, {}
        for e in range(self.m_lo + 5, self.m_hi + 1): P6[e] = self.price(self.a.get(e, 6)).reindex(idx)
        for e in self.ends:
            w = self.a.get(e)
            P[e], S[e] = self.price(w).reindex(idx), self.seoul(w)
            PG[e] = self.price(w, '자치구')
            CNT[e] = w.groupby('동').size().reindex(idx).fillna(0)
            wj = self.wj.get(e); g = wj.groupby('동')['y'].agg(['median', 'size']); PJ[e] = g['median'].where(g['size'] >= 10).reindex(idx)
            ws = self.ws.get(e); g = ws.groupby('동')['평당 금액(만원)'].agg(['median', 'size']); PS[e] = g['median'].where(g['size'] >= 10).reindex(idx)
            ws6 = self.ws.get(e, 6); g = ws6.groupby('동')['평당 금액(만원)'].agg(['median', 'size']); PS6[e] = g['median'].where(g['size'] >= 10).reindex(idx)
            S6[e] = self.seoul(self.a.get(e, 6))
            ww_ = self.wr.get(e); g = ww_.groupby('동')['y'].agg(['median', 'size']); PW[e] = g['median'].where(g['size'] >= 10).reindex(idx)
        self.P, self.S, self.P6 = P, S, P6
        F = {}
        for e in self.ends:
            w = self.a.get(e); Y = self.Y
            g = w.groupby('동')
            f = pd.DataFrame(index=idx)
            f[FLOOR], f[AGE], f[AREA] = g['층'].mean().reindex(idx), g['건축년도'].mean().reindex(idx), g['임대면적(㎡)'].mean().reindex(idx)
            f[NEW] = g['신축'].mean().reindex(idx) * 100
            sd = g[Y].std().reindex(idx); mu = g[Y].mean().reindex(idx); f[CV] = sd / mu * 100
            f[M6] = (P6[e] / P6[e - 6] - 1) * 100 if (e - 6) in P6 else np.nan
            f[M12] = (P[e] / P[e - 12] - 1) * 100 if (e - 12) in P else np.nan
            f[G24] = (P[e] / P[e - 24] - 1) * 100 if (e - 24) in P else np.nan
            f[REL] = P[e] / S[e]
            f[SEOUL12] = (S[e] / S[e - 12] - 1) * 100 if (e - 12) in S else np.nan
            f[GU12] = ((PG[e] / PG[e - 12] - 1) * 100).reindex(d['자치구']).values if (e - 12) in PG else np.nan
            f[VOL] = np.log1p(CNT[e])
            f[VOLCHG] = (CNT[e] / CNT[e - 12].replace(0, np.nan) - 1) * 100 if (e - 12) in CNT else np.nan
            f[RT] = self.RATE.get(e, np.nan)
            f[RTCHG] = self.RATE.get(e, np.nan) - self.RATE.get(e - 12, np.nan)
            f[SALE12] = (PS[e] / PS[e - 12] - 1) * 100 if (e - 12) in PS else np.nan
            f[JR] = PJ[e] / PS[e] * 100
            ww = self.wall.get(e); wsh = ww.groupby('동')['월세계약'].agg(['mean', 'size']); f[WSH] = (wsh['mean'].where(wsh['size'] >= 20) * 100).reindex(idx)
            f[J12] = (PJ[e] / PJ[e - 12] - 1) * 100 if (e - 12) in PJ else np.nan
            f[GUREL] = P[e] / PG[e].reindex(d['자치구']).values
            f[ACC] = 2 * f[M6] - f[M12]
            pk = [P[k] for k in range(e - 36, e + 1, 6) if k in P]
            f[PEAK] = (P[e] / pd.concat(pk, axis=1).max(axis=1) - 1) * 100 if len(pk) >= 4 else np.nan
            f[SEOUL6] = (S6[e] / S6[e - 6] - 1) * 100 if (e - 6) in S6 else np.nan
            f[SEOUL24] = (S[e] / S[e - 24] - 1) * 100 if (e - 24) in S else np.nan
            f[SALE6] = (PS6[e] / PS6[e - 6] - 1) * 100 if (e - 6) in PS6 else np.nan
            jr_now = PJ[e] / PS[e] * 100; jr_old = PJ[e - 12] / PS[e - 12] * 100 if (e - 12) in PS else np.nan
            f[JRCHG] = jr_now - jr_old
            f[W12] = (PW[e] / PW[e - 12] - 1) * 100 if (e - 12) in PW else np.nan
            F[e] = f
        for e in self.ends:                                      # 주변 5개 동 12개월 상승률 평균 (자기 동 제외)
            v = F[e][M12].values; F[e][NB12] = np.nanmean(np.where(np.isnan(v[self.nb]), np.nan, v[self.nb]), axis=1) if True else np.nan
        pm, kp = self.pm, self.kp
        gu = d['자치구'].astype(str).str.strip().values
        for e in self.ends:
            ap = pm[(pm['ma'] > e - 36) & (pm['ma'] <= e)].groupby('구')['총_세대_수'].sum()
            st_ = pm[(pm['ms'] > e - 36) & (pm['ms'] <= e)].groupby('구')['총_세대_수'].sum()
            pp = pm[(pm['ms'] > e - 84) & (pm['ms'] <= e) & ~(pm['mc'] <= e)].groupby('구')['총_세대_수'].sum()
            dn = pm[(pm['mc'] > e - 12) & (pm['mc'] <= e)].groupby('구')['총_세대_수'].sum()
            F[e][GAPP], F[e][GSTART], F[e][GPIPE], F[e][GDONE] = [pd.Series(gu).map(sr).fillna(0).values for sr in (ap, st_, pp, dn)]
            F[e][DDONE] = kp[(kp['m'] > e - 36) & (kp['m'] <= e)].groupby('행정동_코드')['세대'].sum().reindex(idx).fillna(0).values
        self.F = F

    def feats(self, e):
        df = self.F[e].join(self.static)
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

    def quick(self, feat):
        """항목 고르기용: 개발 구간 시험만, 기간별 모델 오차율·R² 평균"""
        out = {}
        for h in STARTS:
            res = self.tests(feat, h, DEV_LAST[h])[0]
            m = res[res['방법'] == '예측 모델'][['오차율 중앙(%)', '결정계수 R²']].mean()
            out[h] = (m['오차율 중앙(%)'], m['결정계수 R²'])
        return out

    def halves(self, feat):
        """모든 시험을 돌려 기간별·구간별(고른 구간/뒤 구간) 모델 오차율과 R² 평균"""
        out = {}
        for h in (12, 36):
            res = self.tests(feat, h)[0]; m = res[res['방법'] == '예측 모델']
            for half, sel in [('dev', m['시험연도'] <= DEV_LAST[h]), ('hold', m['시험연도'] > DEV_LAST[h])]:
                mm = m[sel][['오차율 중앙(%)', '결정계수 R²']].mean(); out[(h, half)] = (mm['오차율 중앙(%)'], mm['결정계수 R²'])
        return out

    def report(self, feat, tag, label):
        kind, Y = self.kind, self.Y
        title = {'jeonse': '전세 평당 보증금', 'wolse': '월세 계약 평당 월세', 'dep': '월세 계약 평당 보증금'}[kind]
        pre = {'jeonse': '아파트전세', 'wolse': '아파트월세', 'dep': '아파트월세보증금'}[kind]
        print(f'===== {label} {tag} | {title} | 항목 {len(feat)}개: {feat}')
        order = ['그대로', '서울 변화율 규칙', '예측 모델']; col = ['오차율 중앙(%)', '10% 이내(%)', '평균 오차(만원/평)', '결정계수 R²']
        allres = []
        for h in STARTS:
            nm = NAMES[h]; res, (ym, model, te) = self.tests(feat, h); allres.append(res)
            avg = res.groupby('방법')[col].mean().round(3).reindex(order)
            print(); print(f'[{nm} 뒤: 시험 {res["시험"].nunique()}번 평균]'); print(avg.to_string())
            dev = res[res['시험연도'] <= DEV_LAST[h]].groupby('방법')[['오차율 중앙(%)', '결정계수 R²']].mean().round(3).reindex(order)
            hold = res[res['시험연도'] > DEV_LAST[h]].groupby('방법')[['오차율 중앙(%)', '결정계수 R²']].mean().round(3).reindex(order)
            print(f'  (항목 고를 때 쓴 구간 {int(res[res["시험연도"] <= DEV_LAST[h]]["시험연도"].nunique())}번 / 확인용 구간 {int(res[res["시험연도"] > DEV_LAST[h]]["시험연도"].nunique())}번)')
            print('  확인용 구간:', ' | '.join(f'{k} {v["오차율 중앙(%)"]}%/{v["결정계수 R²"]}' for k, v in hold.iterrows()))
            pi = permutation_importance(model, te[feat], RATIO(te), n_repeats=5, random_state=0)
            imp = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)
            print(f'[{nm} 뒤 항목 중요도 ({ym}년 시험 구간)]'); print(imp.head(8).round(3).to_string())
        pd.concat(allres).drop(columns=['시험연도']).to_csv(OUT + f'{pre}_{tag}_시험별결과.csv', index=False, encoding='utf-8-sig')
        now = self.feats(self.m_hi)
        cur = f'현재 평당 {"월세" if kind == "wolse" else "보증금"}(최근12개월 ' + ('실거래' if kind == 'jeonse' else '구성 고정 가격') + ', 만원)'
        r = 2 if kind == 'wolse' else 0
        out = pd.DataFrame({'자치구': self.d['자치구'], '행정동': self.d['행정동'], cur: now['현재 가격'].round(r)})
        print(); print(f'===== {tag} 1년·2년·3년 뒤 예측 (2026-09 기준) =====')
        for h in STARTS:
            nm = NAMES[h]; D = self.D[h]
            model = xgb().fit(D[feat], RATIO(D))
            pred = (now['현재 가격'] * np.exp(model.predict(now[feat]))).where(now['현재 가격'].notna())
            out[f'{nm} 뒤 예측(만원/평)'] = pred.round(r)
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
    lab = Lab('wolse')
    FEAT = ['6개월 상승률(%)', '12개월 상승률(%)', '상대 가격(서울 대비)', '자치구 착공 후 미준공 세대수', '자치구 최근3년 허가 세대수', '자치구 최근1년 준공 세대수']
    lab.report(FEAT, 'v15', '월세')
