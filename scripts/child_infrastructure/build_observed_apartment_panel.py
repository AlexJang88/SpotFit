"""서울 상권분석서비스의 실제 분기 관측값으로 연도별 행정동 아파트 패널을 만든다."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
def read_csv(path:Path)->pd.DataFrame:
    for enc in ('utf-8-sig','cp949','utf-8'):
        try:return pd.read_csv(path,encoding=enc,dtype=str)
        except UnicodeDecodeError:continue
    raise UnicodeError(path)
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'data/raw/서울시 상권분석서비스(아파트-행정동).csv');p.add_argument('--output',type=Path,default=ROOT/'data/interim/child_housing_observed');a=p.parse_args()
    source=a.input if a.input.is_absolute() else ROOT/a.input;out=a.output if a.output.is_absolute() else ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    d=read_csv(source);d=d[d['기준_년분기_코드'].str.match(r'^202[1-5]4$')].copy();d['year']=d['기준_년분기_코드'].str[:4].astype(int);d['dong_code']=d['행정동_코드'].str.zfill(8)+'00'
    numeric=[c for c in d.columns if c not in ['기준_년분기_코드','행정동_코드','행정동_코드_명','year','dong_code']]
    for c in numeric:d[c]=pd.to_numeric(d[c].str.replace(',','',regex=False),errors='coerce')
    area_bins=[c for c in numeric if c.startswith('아파트_면적_') and c.endswith('_세대_수')];price_bins=[c for c in numeric if c.startswith('아파트_가격_') and c.endswith('_세대_수')]
    d['observed_apartment_households_by_area']=d[area_bins].sum(axis=1,min_count=1);d['observed_apartment_households_by_price']=d[price_bins].sum(axis=1,min_count=1)
    if d.duplicated(['dong_code','year']).any():raise ValueError('행정동·연도 중복')
    d.to_csv(out/'observed_apartment_panel_2021_2025.csv',index=False,encoding='utf-8-sig')
    q=d.groupby('year').agg(dongs=('dong_code','nunique'),rows=('dong_code','size'),missing_avg_price=('아파트_평균_시가',lambda x:x.isna().sum()),missing_households=('observed_apartment_households_by_area',lambda x:x.isna().sum())).reset_index();q.to_csv(out/'quality.csv',index=False,encoding='utf-8-sig');print(q.to_string(index=False));print(out)
if __name__=='__main__':main()
