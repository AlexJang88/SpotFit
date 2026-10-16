# 로컬 데이터 분석 환경

## 설치 목적별 구분

| 파일 | 용도 | 설치 위치 |
| --- | --- | --- |
| `data_analyze/requirements.txt` | CSV 전처리, 통계 분석, 시각화 | 개발 PC의 분석용 가상환경 |
| `scripts/upload-requirements.txt` | 데이터 업로드 도구 | 업로드를 실행하는 환경 |
| `ml/requirements.txt` | 모델 학습/추론 공통 패키지 | 백엔드 및 Airflow 이미지 |
| `backend/requirements.txt` | API 서버 패키지 | 백엔드 이미지 |

프로젝트 루트에서 실행합니다. 서버와 맞추려면 Python 3.12를 사용하세요.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r data_analyze/requirements.txt
```

### Linux

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r data_analyze/requirements.txt
```

Python 3.12가 없다면 먼저 설치하거나 사용 가능한 Python으로 가상환경을 만드세요.
IDE의 Python 인터프리터도 해당 가상환경으로 선택해야 합니다.
업로드가 필요하면 같은 환경에 `scripts/upload-requirements.txt`를 추가 설치합니다.
로컬 모델 학습까지 필요할 때만 `ml/requirements.txt`를 추가 설치합니다.
서버 코드에서 그래프를 생성하게 되면 해당 의존성을 서버 requirements에도 선언해야 합니다.
기존 설치 패키지는 자동 제거되지 않으며, 서버 이미지는 재빌드해야 변경이 반영됩니다.
