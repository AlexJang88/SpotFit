"""연도 t 특징과 t+1/t+3 아동 인구 증감률을 분리한 누수 없는 예측 패널을 생성한다."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
GROUPS=['infant','elementary','middle','high']

def main()->None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--population',type=Path,required=True); p.add_argument('--housing',type=Path,required=True); p.add_argument('--output',type=Path,default=ROOT/'data/interim/child_prediction_panel'); a=p.parse_args()
    pp=a.population if a.population.is_absolute() else ROOT/a.population; hp=a.housing if a.housing.is_absolute() else ROOT/a.housing; out=a.output if a.output.is_absolute() else ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
    pop=pd.read_csv(pp,dtype={'dong_code':str}); pop['region_key']=pop.gu.astype(str).str.strip()+'|'+pop.dong.astype(str).str.strip()
    housing=pd.read_csv(hp,dtype={'dong_code':str})
    data=pop.merge(housing.drop(columns=['행정동_코드_명'],errors='ignore'),on=['dong_code','year'],how='left',validate='one_to_one',indicator=True)
    data[data['_merge'].ne('both')].to_csv(out/'unmatched_observed_housing.csv',index=False,encoding='utf-8-sig');data=data.drop(columns='_merge').sort_values(['dong_code','year'])
    data['observed_apartment_household_yoy']=data.groupby('dong_code').observed_apartment_households_by_area.pct_change(fill_method=None).replace([float('inf'),float('-inf')],pd.NA)
    for horizon in (1,3):
        future=data[['region_key','year',*[f'children_{g}' for g in GROUPS]]].copy(); future['year']=future.year-horizon
        future=future.rename(columns={f'children_{g}':f'future_{horizon}y_children_{g}' for g in GROUPS})
        data=data.merge(future,on=['region_key','year'],how='left',validate='one_to_one')
        for g in GROUPS:
            current=data[f'children_{g}']; target=data[f'future_{horizon}y_children_{g}']
            data[f'target_{horizon}y_growth_{g}']=(target-current).div(current.where(current.gt(0)))
    data.to_csv(out/'child_prediction_panel.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for h in (1,3):
        for g in GROUPS:
            col=f'target_{h}y_growth_{g}'; rows.append({'horizon_years':h,'age_group':g,'trainable_rows':int(data[col].notna().sum()),'years':','.join(map(str,sorted(data.loc[data[col].notna(),'year'].unique())))})
    pd.DataFrame(rows).to_csv(out/'target_coverage.csv',index=False,encoding='utf-8-sig'); print(pd.DataFrame(rows).to_string(index=False)); print(out)
if __name__=='__main__': main()
