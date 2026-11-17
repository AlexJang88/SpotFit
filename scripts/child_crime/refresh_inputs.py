"""Strict adapters for downloaded wide statistical CSVs into existing feature schemas."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def household_frame(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    mappings = {('총가구','소계','가구수 (가구)','소계'): 'households',
                ('일반가구','소계','가구수 (가구)','소계'): 'ordinary_households',
                ('일반가구','소계','가구원수 (명)','소계'): 'ordinary_members',
                ('집단가구','집단시설가구','가구원수 (명)','소계'): 'institution_members',
                ('외국인가구','소계','가구수 (가구)','소계'): 'foreign_households'}
    frame['field'] = [mappings.get(tuple(r)) for r in frame[
        ['가구형태별(1)','가구형태별(2)','가구원별(1)','가구원별(2)']].to_numpy()]
    result = wide_to_long(frame, '동별(2)')
    if not set(mappings.values()).issubset(result):
        raise ValueError('Household structure schema changed')
    return result


def births_frame(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    if not frame['성별(1)'].eq('합계').all():
        raise ValueError('Birth/death totals required')
    frame['field'] = frame['항목'].map({'출생자수': 'births', '사망자수': 'deaths'})
    months = [c for c in frame if re.fullmatch(r'20\d\d\.\s*\d{2}', c)]
    long = frame.melt(id_vars=['동별(2)', 'field'], value_vars=months, var_name='month', value_name='value')
    long['value'] = pd.to_numeric(long.value.str.replace(',', ''), errors='coerce')
    long['year'] = long.month.str[:4].astype(int)
    if long.duplicated(['동별(2)','field','month']).any() or long.field.isna().any():
        raise ValueError('Duplicate/unknown birth/death fields')
    grouped = long.groupby(['year','동별(2)','field']).value
    if not grouped.size().eq(12).all():
        raise ValueError('Birth/death annual totals require all 12 months')
    return grouped.sum(min_count=12).unstack().reset_index().rename(columns={'동별(2)': 'gu'})


def business_frame(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    mapping = {'사업체수': 'businesses', '사업체밀도': 'business_density',
               '종사자밀도(A)': 'worker_density', '평균종사자': 'workers_per_business'}
    frame['field'] = frame['항목'].str.replace(r' \([^)]*\)$', '', regex=True).map(mapping)
    return wide_to_long(frame, '동별(2)')


def student_frame(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    level = frame['항목'].map({'초등학교': 'elem', '중학교': 'middle', '고등학교': 'high'})
    move = frame['구분별(1)'].map({'전입': 'in', '전출': 'out'})
    area = frame['구분별(2)'].map({'도내': 'within', '도외': 'outside'})
    if level.isna().any() or move.isna().any() or area.isna().any():
        raise ValueError('Student movement schema changed')
    frame['field'] = 'student_' + level + '_' + move + '_' + area
    result = wide_to_long(frame, '자치구별(2)')
    for level in ('elem','middle','high'):
        result[f'student_{level}_net'] = (result[f'student_{level}_in_within'] + result[f'student_{level}_in_outside']
                                        - result[f'student_{level}_out_within'] - result[f'student_{level}_out_outside'])
    return result


def wide_to_long(frame, gu_col):
    frame = frame.loc[frame.field.notna()].copy()
    years = [c for c in frame if re.fullmatch(r'20\d\d', c)]
    long = frame.melt(id_vars=[gu_col, 'field'], value_vars=years, var_name='year', value_name='value')
    long['value'] = pd.to_numeric(long.value.str.replace(',', ''), errors='coerce')
    long['year'] = long.year.astype(int)
    result = long.pivot(index=['year',gu_col], columns='field', values='value').reset_index().rename(columns={gu_col:'gu'})
    return result


def merge_complete(existing, incoming):
    from scripts.child_crime.build_borough_experiment_panel import GU
    columns = [c for c in incoming if c not in ('year','gu')]
    if incoming.duplicated(['year','gu']).any():
        raise ValueError('Duplicate statistical keys')
    valid = []
    for year, frame in incoming.groupby('year'):
        if set(frame.gu) != set(GU) or not np.isfinite(frame[columns].to_numpy(float)).all():
            # A new incompletely published year must never replace an existing full year.
            raise ValueError(f'Statistical input {year} incomplete: {columns}')
        valid.append(frame)
    newer = pd.concat(valid).set_index(['year','gu'])
    prior = existing.set_index(['year','gu'])
    common = prior.index.intersection(newer.index)
    # Published statistical exports sometimes round values to two decimal places.
    a,b = prior.loc[common,columns].to_numpy(float), newer.loc[common,columns].to_numpy(float)
    if not np.allclose(a,b,atol=.005,rtol=1e-9,equal_nan=True):
        raise ValueError(f'Historical statistical values revised; manual review required: {columns}')
    return pd.concat([prior,newer.loc[newer.index.difference(prior.index)]]).reset_index()
