import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.child_crime import run_refresh as refresh
from scripts.child_crime import collect_refresh as collection
from app.services import child_crime_exposure as service


def test_preflight_detects_stale_image_before_preprocessing(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util,'find_spec',lambda name: None if name=='openpyxl' else object())
    with pytest.raises(RuntimeError,match='Rebuild the backend image'):
        refresh.preflight()


def test_merge_preserves_history_and_adds_new_year():
    old = pd.DataFrame({'year': [2024], 'value': [3]})
    new = pd.DataFrame({'year': [2024,2025], 'value': [3,4]})
    assert refresh.merge_download(old,new,['year'],['value']).value.tolist() == [3,4]


def test_merge_rejects_revision():
    with pytest.raises(ValueError, match='historical'):
        refresh.merge_download(pd.DataFrame({'year':[2024],'v':[3]}),pd.DataFrame({'year':[2024],'v':[4]}),['year'],['v'])


def test_merge_rejects_duplicates():
    d = pd.DataFrame({'year':[2024,2024],'v':[3,3]})
    with pytest.raises(ValueError, match='duplicate'):
        refresh.merge_download(d.iloc[:1],d,['year'],['v'])


def crime_frame():
    from scripts.child_crime.build_borough_experiment_panel import GU
    return pd.DataFrame([{'년도':2025,'시군구':g,'범죄':c,'발생건수':1.,'체포수':1.}
                         for g in GU for c in ('강간·강제추행','강도','살인','절도','폭력')])


def test_crime_complete_grid():
    refresh.validate_crime(crime_frame())


def test_crime_missing_borough_rejected():
    with pytest.raises(ValueError, match='25 boroughs'):
        refresh.validate_crime(crime_frame().iloc[:-1])


def test_new_missing_counts_rejected():
    d = crime_frame(); d.loc[0,'발생건수'] = np.nan
    with pytest.raises(ValueError, match='Unapproved'):
        refresh.validate_crime(d)
    row = d.iloc[0]
    refresh.validate_crime(d, {(2025,row['시군구'],row['범죄'],'발생건수')})


def test_negative_count_rejected():
    d=crime_frame(); d.loc[0,'발생건수']=-1
    with pytest.raises(ValueError, match='negative'):
        refresh.validate_crime(d)


def test_atomic_pointer_resolves_both_horizons(tmp_path,monkeypatch):
    monkeypatch.setattr(service,'MODEL_ROOT',tmp_path)
    assert service.required_paths(2)[0].parent == tmp_path/'2y'
    refresh.write_json(tmp_path/'current_release.json',{'release':'20261117T010000Z'})
    for h in (2,3):
        assert service.required_paths(h)[0].parent == tmp_path/'releases/20261117T010000Z'/f'{h}y'


@pytest.mark.parametrize('release',['../bad','C:/bad','',None])
def test_pointer_path_traversal_rejected(tmp_path,monkeypatch,release):
    monkeypatch.setattr(service,'MODEL_ROOT',tmp_path)
    refresh.write_json(tmp_path/'current_release.json',{'release':release})
    with pytest.raises(service.ChildCrimeExposureError):
        service.required_paths(2)


def test_input_gate_missing_feature():
    config={'two':{'features':['x']},'old_crime':{'features':[]},'base':{'population_features':[],
            'new_crime_winner':{'features':[]}},'three':{'direct_features':[]}}
    frame=pd.DataFrame({'year':[2025]*25,'gu':list(map(str,range(25)))})
    with pytest.raises(ValueError,match='missing features'):
        refresh.input_gate(frame,config)


def test_csv_download_uses_korean_columns_not_preview(monkeypatch):
    calls=[]
    meta={'metaTypeAllList':[{'colNm':'yr','colKorNm':'년도'}],
          'datasetDetail':{'trgtTableIndctNm':'crime','clctCycle':'annual'}}
    def fetch(request):
        calls.append(request)
        return json.dumps(meta).encode() if isinstance(request,str) else '년도\n2025\n'.encode('cp949')
    monkeypatch.setattr(collection,'fetch',fetch)
    data,info=collection.download(481)
    assert info['rows']==1
    assert 'dataCsvDown.do' in calls[1].full_url
    assert b'tdColNmArr=%EB%85%84%EB%8F%84' in calls[1].data
    assert data.decode('utf-8-sig').startswith('년도')


def test_empty_download_rejected(monkeypatch):
    meta={'metaTypeAllList':[{'colKorNm':'년도'}], 'datasetDetail':{}}
    monkeypatch.setattr(collection,'fetch',lambda r: json.dumps(meta).encode() if isinstance(r,str) else b'\r\n')
    with pytest.raises(ValueError,match='empty-response'):
        collection.download(481)


def test_refresh_api_auth(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.controllers.model_bootstrap_controller import router
    from app.services import crime_refresh_job
    app=FastAPI(); app.include_router(router,prefix='/api')
    client=TestClient(app)
    monkeypatch.setenv('MODEL_TRAINING_API_KEY','test-admin-only')
    monkeypatch.setattr(crime_refresh_job,'start_refresh',lambda:{'state':'running'})
    assert client.post('/api/models/crime/refresh').status_code==403
    assert client.get('/api/models/crime/refresh').status_code==403
    assert client.post('/api/models/crime/refresh',headers={'X-Model-Admin-Key':'test-admin-only'}).status_code==202
    assert client.post('/api/models/crime/refresh',headers={'X-Model-Admin-Key':'test-admin-only'},
                       json={'collection_run':'../bad'}).status_code==422
    received=[]
    monkeypatch.setattr(crime_refresh_job,'start_refresh',lambda run: received.append(run) or {'state':'running'})
    assert client.post('/api/models/crime/refresh',headers={'X-Model-Admin-Key':'test-admin-only'},
                       json={'collection_run':'20261117T010000Z'}).status_code==202
    assert received==['20261117T010000Z']


def test_bootstrap_recognizes_promoted_release(tmp_path):
    from app.services import service_model_bootstrap as boot
    root=tmp_path/'models/recommender/v1/child_crime_exposure'
    refresh.write_json(root/'current_release.json',{'release':'candidate1'})
    for h in (2,3):
        core=service.required_paths(h,root)
        for p in (*core,core[0].parent/'feature_score_formula.csv',core[0].parent/'feature_score_breakdown.csv'):
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'artifact')
    assert boot.missing('child_crime',tmp_path)==[]
    core[0].unlink()
    assert len(boot.missing('child_crime',tmp_path))==1


def test_corrupt_pointer_has_safe_bootstrap_status(tmp_path):
    from app.services import service_model_bootstrap as boot
    refresh.write_json(tmp_path/'models/recommender/v1/child_crime_exposure/current_release.json',{})
    assert 'invalid pointer' in boot.missing('child_crime',tmp_path)[0]


def test_births_require_twelve_months(tmp_path):
    from scripts.child_crime.refresh_inputs import births_frame
    frame=pd.DataFrame([{'동별(2)':'종로구','성별(1)':'합계','항목':kind,
                         **{f'2025. {m:02}':'1' for m in range(1,13)}} for kind in ('출생자수','사망자수')])
    path=tmp_path/'births.csv';frame.to_csv(path,index=False,encoding='utf-8-sig')
    result=births_frame(path)
    assert result.births.tolist()==[12]
    assert result.deaths.tolist()==[12]
    frame.drop(columns=['2025. 12']).to_csv(path,index=False,encoding='utf-8-sig')
    with pytest.raises(ValueError,match='12 months'):
        births_frame(path)


def test_statistical_merge_requires_all_boroughs():
    from scripts.child_crime.refresh_inputs import merge_complete
    d=pd.DataFrame({'year':[2025],'gu':['종로구'],'births':[12]})
    with pytest.raises(ValueError,match='incomplete'):
        merge_complete(d,d)


def test_licences_paginate_filter_and_resume(tmp_path,monkeypatch):
    from scripts.child_crime import collect_licences as licences
    monkeypatch.setattr(licences,'SEOUL_AUTHORITIES',('3000000',))
    monkeypatch.setattr(licences.time,'sleep',lambda _:None)
    calls=[]
    def fetch(kind,code,page,key):
        calls.append(page)
        return {'totalCount':2,'items':{'item':[{'OPN_ATMY_GRP_CD':code,'MNG_NO':str(page)}]}}
    monkeypatch.setattr(licences,'fetch',fetch)
    first=licences.collect('entertainment_bars',tmp_path,'fake-secret')
    assert first['rows']==2 and calls==[1,2]
    calls.clear()
    again=licences.collect('entertainment_bars',tmp_path,'fake-secret')
    assert again['requests']==0 and calls==[]
    assert not any('fake-secret' in p.read_text(encoding='utf-8-sig') for p in tmp_path.rglob('*') if p.is_file())


def test_licences_reject_ignored_filter(tmp_path,monkeypatch):
    from scripts.child_crime import collect_licences as licences
    monkeypatch.setattr(licences,'SEOUL_AUTHORITIES',('3000000',))
    monkeypatch.setattr(licences,'fetch',lambda *a:{'totalCount':1,'items':{'item':[{'OPN_ATMY_GRP_CD':'999','MNG_NO':'a'}]}})
    with pytest.raises(ValueError,match='filter/schema'):
        licences.collect('entertainment_bars',tmp_path,'fake-secret')
    assert not list(tmp_path.glob('*.csv'))


def test_licence_errors_do_not_leak_key(monkeypatch):
    from urllib.error import HTTPError
    from scripts.child_crime import collect_licences as licences
    def denied(request,**kwargs):
        raise HTTPError(request.full_url,403,'forbidden',{},None)
    monkeypatch.setattr(licences,'urlopen',denied)
    with pytest.raises(RuntimeError,match='403') as error:
        licences.fetch('singing_bars','3000000',1,'NEVER_PRINT_THIS')
    assert 'NEVER_PRINT_THIS' not in str(error.value)
