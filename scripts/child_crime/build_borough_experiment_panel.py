"""Aggregate user-supplied dong sources to boroughs without duplicating totals."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from inspect_borough_inputs import ROOT, RAW, csvread, spreadsheet

OUT=ROOT/'artifacts/child_crime_borough_v2'
GU=['종로구','중구','용산구','성동구','광진구','동대문구','중랑구','성북구','강북구','도봉구','노원구','은평구','서대문구','마포구','양천구','강서구','구로구','금천구','영등포구','동작구','관악구','서초구','강남구','송파구','강동구']
def num(x):
    return pd.to_numeric(str(x).replace(',',''),errors='coerce')
def wide_xml(path, selector, start):
    rows=spreadsheet(path); output=[]
    for row in rows[2:]:
        if not row or row[0] not in GU: continue
        field=selector(row)
        if field is None: continue
        for k in range(start,min(len(row),len(rows[1]))):
            m=re.match(r'(20\d\d)',rows[1][k])
            if m: output.append([int(m[1]),row[0],field,num(row[k])])
    d=pd.DataFrame(output,columns=['year','gu','field','value'])
    if d.duplicated(['year','gu','field']).any():
        raise ValueError(f'Duplicate observations {path}')
    return d.pivot(index=['year','gu'],columns='field',values='value').reset_index()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=[]
    frames=[]
    for path in sorted((RAW/'연령별 인구수').glob('*.csv')):
        year=int(path.name[:4]); d=csvread(path)
        names=d['행정구역'].str.extract(r'서울특별시\s+(.+?구)\s+([^()]+)\s*\(')
        # Aggregate dong rows only; never add published borough totals again.
        d['gu']=names[0]; d=d[d.gu.isin(GU)].copy()
        a=np.column_stack([d[f'{year}년_계_{age}세'].map(num) for age in range(100)])
        d['child']=a[:,:18].sum(1);d['age0_5']=a[:,:6].sum(1)
        d['age6_12']=a[:,6:13].sum(1);d['age13_17']=a[:,13:18].sum(1)
        d['elderly']=a[:,65:].sum(1)+d[f'{year}년_계_100세 이상'].map(num).to_numpy()
        d['population']=d[f'{year}년_계_총인구수'].map(num)
        part=d.groupby('gu')[['child','age0_5','age6_12','age13_17','elderly','population']].sum()
        assert len(part)==25
        # Confirm aggregated dong totals agree with published borough totals.
        whole=csvread(path)
        whole['gu']=whole['행정구역'].str.extract(r'^서울특별시\s+([^ ]+구)\s*\(')
        whole=whole[whole.gu.isin(GU)].set_index('gu')
        totals=whole[f'{year}년_계_총인구수'].map(num)
        assert np.array_equal(part.population.sort_index(),totals.sort_index()),year
        part['year']=year;frames.append(part.reset_index())
    panel=pd.concat(frames,ignore_index=True)
    crime=csvread(RAW/'서울시_5대범죄_2014_2025_원자료.csv')
    crime['발생건수']=crime['발생건수'].map(num)
    c=crime.groupby(['년도','시군구'])['발생건수'].sum().rename('crime').reset_index().rename(columns={'년도':'year','시군구':'gu'})
    panel=panel.merge(c,on=['year','gu'],how='left',validate='one_to_one')
    panel['log_crime']=np.log1p(panel.crime);panel['log_child']=np.log1p(panel.child)
    panel['exposure']=panel.crime/panel.child*10000;panel['y']=np.log1p(panel.exposure)
    for col in ['child','age0_5','age6_12','age13_17','elderly']:
        panel[col+'_share']=panel[col]/panel.population
    for name in ['유흥주점','단란주점']:
        d=csvread(next((RAW/'유흥주점·단란주점').glob(f'*{name}*.csv')),dtype=str).fillna('')
        assert not d['관리번호'].duplicated().any()
        d['gu']=d['지번주소'].str.extract(r'서울특별시\s+(\S+구)')
        d['gu']=d.gu.fillna(d['도로명주소'].str.extract(r'서울특별시\s+(\S+구)')[0])
        opening=pd.to_datetime(d['인허가일자'].str.strip(),errors='coerce')
        closing=pd.to_datetime(d['폐업일자'].str.strip(),errors='coerce')
        uncertain=closing.isna() & ~d['영업상태명'].isin(['영업/정상'])
        valid=d.gu.isin(GU)&opening.notna()&~uncertain
        valid &= closing.isna() | (closing>=opening)
        audit.append({'source':name,'rows':len(d),'unknown_end_status':int(uncertain.sum()),'excluded_total':int((~valid).sum()),'historical_address_limitation':'current recorded address; relocations/historical suspensions unavailable'})
        observations=[]
        for year in range(2010,2026):
            end=pd.Timestamp(year,12,31)
            counts=d[valid&(opening<=end)&(closing.isna()|(closing>end))].groupby('gu').size().reindex(GU,fill_value=0)
            observations.extend({'year':year,'gu':g,name:count} for g,count in counts.items())
        panel=panel.merge(pd.DataFrame(observations),on=['year','gu'],validate='one_to_one')
    for col in ['유흥주점','단란주점']:
        panel[col+'_per10k']=panel[col]/panel.population*10000
    worker_map={'총종사자수':'workers','자영업주 및 무급가족':'self_family','상용근로자':'regular','임시 및 일용근로자':'temporary','기타 종사자':'other_workers'}
    w=wide_xml(next((RAW/'종사상지위별 종사자수').glob('*.xls')),lambda r:worker_map.get(r[1]),4)
    for col in list(worker_map.values())[1:]:w[col+'_share']=w[col]/w.workers
    panel=panel.merge(w,on=['year','gu'],how='left',validate='one_to_one')
    densmap={'사업체수':'businesses','사업체밀도':'business_density','종사자밀도(A)':'worker_density','평균종사자':'workers_per_business'}
    b=wide_xml(RAW/'사업체 밀도와 조직 형태/사업체 및 종사자 밀도.xls',lambda r:densmap.get(r[1]),3)
    panel=panel.merge(b,on=['year','gu'],how='left',validate='one_to_one')
    hhmap={'총가구_가구[가구]':'households','일반가구_가구[가구]':'ordinary_households','일반가구_가구원_계[명]':'ordinary_members','집단가구_집단시설가구_가구원_계[명]':'institution_members','외국인가구_가구[가구]':'foreign_households'}
    hh=wide_xml(next((RAW/'1인 가구').glob('*.xls')),lambda r:hhmap.get(r[1]),3)
    hh['household_size']=hh.ordinary_members/hh.ordinary_households
    hh['ordinary_share']=hh.ordinary_households/hh.households
    hh['foreign_household_share']=hh.foreign_households/hh.households
    panel=panel.merge(hh,on=['year','gu'],how='left',validate='one_to_one')
    audit.append({'source':'1인 가구','actual_content':'가구형태별 가구 및 가구원; 1인가구 수 없음','used':'자치구 공표 합계; 가구구조로 명명'})
    # Organization export omits parent categories. Do not guess row order semantics.
    audit.append({'source':'사업체현황(조직형태별동별).xls','used':False,'reason':'조직형태 상위 항목명이 누락되어 사업체수/종사자수만 반복됨. 조직형태는 MDIS 명시적 코드로 대체'})
    mig=csvread(next((RAW/'인구이동').glob('*.csv')))
    mig=mig[(mig['성별명']=='계')&mig['구명'].isin(GU)].copy()
    mig['year']=pd.to_numeric(mig['통계연도']);mig['value']=mig['인구수'].map(num)
    # Use the published borough total, validating a single row for each movement type.
    total=mig[mig['동명']=='계'].copy()
    if total.empty: total=mig[mig['동명']=='합계'].copy()
    if total.empty: raise ValueError('No identifiable migration borough totals')
    assert not total.duplicated(['year','구명','이동유형명']).any()
    mt=total.pivot(index=['year','구명'],columns='이동유형명',values='value').reset_index().rename(columns={'구명':'gu'})
    print('MIGRATION_TYPES',mt.columns.tolist(),flush=True)
    mt=mt.rename(columns={col:'migration_'+col for col in mt.columns if col not in ['year','gu']})
    panel=panel.merge(mt,on=['year','gu'],how='left',validate='one_to_one')
    births=[]
    for path in sorted((RAW/'출생사망인구').glob('*.xls')):
        rows=spreadsheet(path)
        for row in rows[2:]:
            if len(row)<5 or row[0] not in GU or row[1]!='합계':continue
            col={'출생자수':'births','사망자수':'deaths'}.get(row[2])
            if not col:continue
            for k in range(4,len(row)):
                year=int(rows[1][k][:4]);v=num(row[k])
                births.append([year,row[0],col,v])
    bd=pd.DataFrame(births,columns=['year','gu','field','value'])
    grouped=bd.groupby(['year','gu','field']).value
    assert (grouped.size()==12).all()
    bd=grouped.sum(min_count=12).unstack().reset_index()
    panel=panel.merge(bd,on=['year','gu'],how='left',validate='one_to_one')
    for col in ['births','deaths',*mt.columns[2:]]:
        panel[col+'_rate']=panel[col]/panel.population*1000
    panel['natural_growth']=(panel.births-panel.deaths)/panel.population

    # Count establishment records by broad industry only; redacted worker counts
    # and changed employee-column layouts are never interpreted as zero.
    cache=OUT/'mdis_borough_counts.csv'
    if cache.exists(): mdis=pd.read_csv(cache)
    else:
        book=next((RAW/'집객시설').glob('*.xlsx'))
        codes=pd.read_excel(book,sheet_name='행정구역',header=3,dtype=str)
        codes=codes[codes.CITY_CD=='11']
        code_map=dict(zip(codes.SIGUN_CD.str[-3:],codes.SIGUN_NM))
        results=[]
        for path in sorted((RAW/'집객시설').glob('*.csv')):
            year=int(path.name[:4])
            if year<2014:continue
            indices=[1,2,4,5] if year==2014 else [1,2,7,9,10]
            counts={};org_counts={};n=0
            for chunk in pd.read_csv(path,header=None,usecols=indices,dtype=str,chunksize=250000,encoding='cp949'):
                chunk=chunk[chunk[1]=='11'].copy()
                chunk['gu']=chunk[2].str.zfill(3).map(code_map)
                if chunk.gu.isna().any(): raise ValueError('MDIS unmatched borough code')
                industry_col=4 if year==2014 else 9
                for (g,sector),value in chunk.groupby(['gu',industry_col]).size().items():
                    counts[(g,sector)]=counts.get((g,sector),0)+int(value)
                if year>=2015:
                    for (g,org),value in chunk.groupby(['gu',7]).size().items():
                        org_counts[(g,org)]=org_counts.get((g,org),0)+int(value)
                n+=len(chunk)
            for g in GU:
                row={'year':year,'gu':g}
                for sector in ['G','I','H','P','Q','R','S','C','L']:
                    row['industry_'+sector]=counts.get((g,sector),0)
                row['mdis_total']=sum(v for (gu,_),v in counts.items() if gu==g)
                for code in ['1','2','3','4','5']:
                    row['org_'+code]=org_counts.get((g,code),0) if year>=2015 else np.nan
                results.append(row)
            print('MDIS',year,'Seoul rows',n,flush=True)
        mdis=pd.DataFrame(results);mdis.to_csv(cache,index=False,encoding='utf-8-sig')
    for col in mdis.columns:
        if col.startswith(('industry_','org_')):mdis[col+'_share']=mdis[col]/mdis.mdis_total
    panel=panel.merge(mdis,on=['year','gu'],how='left',validate='one_to_one')
    panel=panel.sort_values(['gu','year']).reset_index(drop=True)
    groups={
        '인구구조':['log_child','child_share','age0_5_share','age6_12_share','age13_17_share','elderly_share'],
        '유흥시설':['유흥주점','단란주점','유흥주점_per10k','단란주점_per10k'],
        '고용구조':['workers','self_family_share','regular_share','temporary_share','other_workers_share'],
        '사업체밀도':list(densmap.values()),
        '가구구조':['households','ordinary_share','household_size','foreign_household_share','institution_members'],
        '인구이동':[c+'_rate' for c in mt.columns if c not in ['year','gu']],
        '출생사망':['births_rate','deaths_rate','natural_growth'],
        '산업조직구조':[c for c in mdis.columns if c.endswith('_share')],
    }
    # All non-crime/population observations are lagged one year to reduce release lag leakage.
    for group in groups:
        if group=='인구구조':continue
        old=groups[group];new=[]
        for c in old:
            dest=c+'_available_lag1';panel[dest]=panel.groupby('gu')[c].shift(1);new.append(dest)
        groups[group]=new
    trends=[]
    for c in ['y','log_crime','log_child']:
        for lag in [1,2]:
            col=f'{c}_lag{lag}';panel[col]=panel.groupby('gu')[c].shift(lag);trends.append(col)
        col=c+'_change';panel[col]=panel[c]-panel[c+'_lag1'];trends.append(col)
    groups['추세']=trends
    assert not panel.duplicated(['gu','year']).any()
    panel.to_csv(OUT/'panel.csv',index=False,encoding='utf-8-sig')
    (OUT/'feature_groups.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    coverage=panel.groupby('year')[[c for cols in groups.values() for c in cols]].count()
    coverage.to_csv(OUT/'feature_coverage.csv',encoding='utf-8-sig')
    print('DONE',panel.shape, 'feature groups',len(groups),flush=True)

if __name__=='__main__':main()
