# 로컬 데이터 분석 환경

## 개발 환경 설치

| 파일 | 용도 | 설치 위치 |
| --- | --- | --- |
| `requirements-dev.txt` | 서버 패키지 + 전처리·통계·시각화·Excel·공간 분석 | 개발 PC 가상환경 |
| `requirements-server.txt` | API·학습·추론·MLflow·MinIO | Docker 서버 이미지 |

프로젝트 루트에서 실행합니다. 서버와 맞추려면 Python 3.12를 사용하세요.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### Linux

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-dev.txt
```

Python 3.12가 없다면 먼저 설치하거나 사용 가능한 Python으로 가상환경을 만드세요.
IDE의 Python 인터프리터도 해당 가상환경으로 선택해야 합니다.
`requirements-dev.txt`가 업로드와 모델 학습 패키지를 모두 포함하므로 추가 requirements 설치는 필요하지 않습니다.
서버에서 실제로 필요한 패키지는 `requirements-server.txt`에, 개발·EDA에만 필요한 패키지는 `requirements-dev.txt`에 추가합니다.
기존 설치 패키지는 자동 제거되지 않으며, 서버 이미지는 재빌드해야 변경이 반영됩니다.
