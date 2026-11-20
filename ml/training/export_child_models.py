"""검증 우승 모델을 README의 MinIO 업로드 입력 구조로 내보낸다.

원격 연결·역직렬화 없이 기존 자체 생성 joblib를 복사한다.
MinIO 객체 버전 ID는 업로드 후에만 알 수 있으므로 manifest를 미리 만들지 않는다.
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import pandas as pd


def export_models(root, result_dir, version):
    # 버전은 경로 구성 요소 하나만 허용하여 상위 경로 접근을 막는다.
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in version):
        raise ValueError('버전에는 영문·숫자·하이픈·밑줄만 사용하세요.')
    comparison=pd.read_csv(result_dir/'comparison.csv')
    winners=pd.read_csv(result_dir/'winners.csv')
    plans=[]
    for row in winners.itertuples():
        if row.age not in ['infant','elementary','middle','high']:
            raise ValueError('알 수 없는 연령군')
        matches=comparison[comparison.run_id==row.run_id]
        if len(matches)!=1:
            raise ValueError('우승 run과 조합 매핑이 유일하지 않습니다.')
        # 원 실험의 조합 생성 순서를 재구성하여 정확한 joblib 파일을 찾는다.
        from itertools import combinations
        rank=pd.read_csv(result_dir/(row.age+'_feature_ranking.csv'))
        top=rank[rank.eligible.astype(str).str.lower()=='true'].head(5).feature.tolist()
        combos=[list(c) for n in range(1,6) for c in combinations(top,n)]
        index=next(i for i,c in enumerate(combos,1) if ' | '.join(c)==row.features)
        source=result_dir/f'{row.age}_{index:02d}.joblib'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        target=root/'models'/('spotfit-child-count-'+row.age)/version/'model.joblib'
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise FileExistsError(f'동일 버전 덮어쓰기 금지: {target}')
        plans.append((source,target,digest,row))
    # 모든 경로·파일을 먼저 검증한 뒤 복사한다. 원본 실험 산출물은 보존한다.
    exports=[]
    for source,target,digest,row in plans:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(source,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        exports.append({'age':row.age,'run_id':row.run_id,'export_path':str(target),
            'sha256':digest,'upload_status':'not_uploaded','minio_version':version})
    pd.DataFrame(exports).to_csv(result_dir/'minio_export.csv',index=False,encoding='utf-8-sig')
    return exports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--version',required=True)
    a=p.parse_args()
    for item in export_models(Path(__file__).resolve().parents[2],a.results,a.version):
        print(item['export_path'])
