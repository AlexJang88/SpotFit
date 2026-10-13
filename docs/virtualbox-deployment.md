# VirtualBox Ubuntu 내부망 테스트 서버

현재 개발 구성을 사내·가정 내부망의 Windows PC 위 VirtualBox Ubuntu VM에서 시험하는 절차입니다.
외부 인터넷에 공개하는 운영 배포 구성이 아닙니다. 프런트엔드 개발 서버와 FastAPI reload가
그대로 실행되므로 신뢰할 수 있는 내부망에서만 사용합니다.

## 권장 VM 구성

- Ubuntu Server 24.04 LTS, 64-bit
- CPU 2코어 이상
- RAM 8GB 이상 권장 (Airflow와 딥러닝 라이브러리 포함)
- 동적 확장 디스크 60GB 이상
- 네트워크: **어댑터에 브리지(Bridged Adapter)**
- Windows의 유선 연결을 사용 중이면 Ethernet, 무선 연결이면 현재 Wi-Fi 어댑터 선택

브리지 모드에서는 VM이 Windows 호스트와 별개의 장치처럼 내부 공유기에서 IP를 받습니다.
내부망의 다른 PC도 그 IP로 접근할 수 있습니다. 회사 네트워크가 새 장치/MAC 주소를 차단하거나
Wi-Fi 브리지를 허용하지 않는 경우에는 네트워크 관리자 확인 또는 NAT 포트 포워딩이 필요합니다.

Ubuntu 설치 시 OpenSSH Server를 선택합니다. 설치 후 VM의 주소를 확인합니다.

```sh
hostname -I
ip -4 addr
```

예를 들어 `192.168.0.3`을 받았다면 공유기의 DHCP 예약 기능으로 VM의 MAC 주소에 이 IP를
예약하는 방식을 권장합니다. 예약할 수 없다면 재부팅 후 IP가 바뀔 수 있습니다.

Windows PowerShell에서 연결을 먼저 확인합니다.

```powershell
ping 192.168.0.3
ssh ubuntu사용자명@192.168.0.3
```

## Docker와 프로젝트 설치

Ubuntu VM에서 Git을 설치합니다.

```sh
sudo apt update
sudo apt install -y git ca-certificates curl
```

Docker Engine과 Compose 플러그인은 Docker 공식 Ubuntu 저장소 설치 절차를 따릅니다.

- https://docs.docker.com/engine/install/ubuntu/

설치 후 현재 사용자를 Docker 그룹에 추가하고 다시 로그인합니다.

```sh
sudo usermod -aG docker "$USER"
exit
```

다시 SSH로 접속한 뒤 확인합니다.

```sh
docker version
docker compose version
```

GitHub 저장소가 공개 저장소라면 다음처럼 받습니다.

```sh
git clone https://github.com/AlexJang88/SpotFit.git
cd SpotFit
```

비공개 저장소라면 서버 전용 읽기 Deploy Key 또는 Fine-grained Personal Access Token을 사용합니다.
토큰을 clone URL에 직접 포함하거나 셸 기록에 남기지 않습니다.

## 내부망 바인딩과 비밀값 설정

Ubuntu 서버 전용 예시에서 환경 파일을 만들고 권한을 제한합니다. Windows 개발용 `.env`와
서버 비밀값을 섞지 않기 위해 서버에서는 `.env.server`를 사용합니다.

```sh
cp .env.server.example .env.server
chmod 600 .env.server
nano .env.server
```

이미 `.env.server`가 있다면 다시 복사하지 말고 편집만 하세요. 이 파일은 Git에서 제외되므로
`git pull`을 해도 새 Airflow·MLflow 항목이 자동 추가되지 않습니다.

VM 주소가 `192.168.0.3`일 때 다음처럼 설정합니다.

```dotenv
APP_BIND_ADDRESS=192.168.0.3
APP_PORT=8080
POSTGRES_PASSWORD=반드시_새로_생성한_값
POSTGRES_BIND_ADDRESS=192.168.0.3
AIRFLOW_BIND_ADDRESS=192.168.0.3
AIRFLOW_ADMIN_USER=spotfit
AIRFLOW_FERNET_KEY=Fernet_키_생성_명령으로_만든_값
AIRFLOW_SECRET_KEY=별도_랜덤_값
MLFLOW_BIND_ADDRESS=192.168.0.3
MINIO_ROOT_USER=spot_fit_admin
MINIO_ROOT_PASSWORD=반드시_새로_생성한_값
```

Airflow 키는 Ubuntu의 Python과 OpenSSL로 생성합니다.

```sh
python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'
openssl rand -hex 32
```

첫 번째 출력은 `AIRFLOW_FERNET_KEY`, 두 번째 출력은 `AIRFLOW_SECRET_KEY`에 입력합니다.

비밀번호 후보는 아래 명령을 두 번 실행해 서로 다른 값으로 생성합니다.

```sh
openssl rand -hex 24
openssl rand -hex 24
```

생성한 값을 `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`에 각각 입력합니다.
`.env.server`는 Git에 커밋하지 않습니다. `.env.server.example`에는 실제 비밀번호를 넣지 않습니다.
Nginx, PostgreSQL, Airflow, MLflow는 VM의 내부망 주소에 바인딩됩니다. PostgreSQL은 내부망에서 DBeaver로
접속할 수 있지만 공유기의 포트 포워딩으로 외부에 공개하지 않습니다. MinIO는
`127.0.0.1:9000-9001`에 유지되어 다른 내부망 PC에서 직접 접근할 수 없습니다.

## 실행과 확인

서버용 override는 VM 재부팅 후 컨테이너가 다시 시작되도록 정책을 추가합니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
```

MinIO 컴파일과 Airflow·PyTorch 학습 이미지의 패키지 다운로드 때문에 최초 빌드에는 시간이 걸리고
여러 GB의 디스크 공간을 사용합니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs -f
```

Ubuntu VM 안에서 확인합니다.

```sh
curl http://127.0.0.1:8080/api/spot_fit
```

Windows와 같은 내부망의 다른 기기에서는 VM 주소로 접속합니다.

- 화면: `http://192.168.0.3:8080`
- API 문서: `http://192.168.0.3:8080/api/docs`
- API 확인: `http://192.168.0.3:8080/api/spot_fit`

접속되지 않으면 Ubuntu에서 `ss -lntp | grep 8080`으로 바인딩을 확인하고, Windows 방화벽,
공유기의 무선 클라이언트 격리(AP isolation), 회사망 정책을 확인합니다. Ubuntu 방화벽을 사용하는
경우 실제 내부망 CIDR만 허용합니다. 예시는 내부망이 `192.168.0.0/24`일 때입니다.

```sh
sudo ufw allow OpenSSH
sudo ufw allow from 192.168.0.0/24 to any port 8080 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 5432 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 8081 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 5000 proto tcp
sudo ufw enable
```

Docker의 공개 포트는 호스트 방화벽 동작과 상호작용할 수 있으므로, 접근 제한의 기준은
`APP_BIND_ADDRESS`를 VM의 내부 IPv4 주소로 지정하는 것입니다. 공유기에서 8080 포트를
포워딩하지 않습니다.

내부망의 브라우저에서는 Airflow `http://192.168.0.3:8081`, MLflow
`http://192.168.0.3:5000`으로 접속합니다. Airflow 계정은 `spotfit`이며 최초 생성된 암호는
다음 명령으로 확인합니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml \
  exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

## DBeaver에서 PostgreSQL 접속

Ubuntu에서 PostgreSQL 포트가 VM 주소에 공개되었는지 확인합니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up -d --force-recreate postgres
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps postgres
sudo ss -lntp | grep 5432
```

Windows PowerShell에서 포트 연결을 확인합니다.

```powershell
Test-NetConnection 192.168.0.3 -Port 5432
```

DBeaver에서 **새 데이터베이스 연결 → PostgreSQL**을 선택하고 다음 값을 입력합니다.

- Host: `192.168.0.3`
- Port: `5432`
- Database: `.env.server`의 `POSTGRES_DB` 값 (`spot_fit`)
- Username: `.env.server`의 `POSTGRES_USER` 값 (`spot_fit`)
- Password: `.env.server`의 실제 `POSTGRES_PASSWORD` 값
- SSL: 내부망 테스트에서는 `disable`

비밀번호가 맞는지 컨테이너 내부에서 TCP 인증으로 확인할 수 있습니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml exec postgres \
  sh -c 'psql -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -W'
```

DB 볼륨이 이미 만들어진 뒤 `.env.server`의 비밀번호만 변경했다면 기존 DB 비밀번호는 자동으로
바뀌지 않습니다. 이 경우 `psql`에서 `\password spot_fit`을 실행하여 같은 비밀번호로 맞춥니다.

## MinIO 콘솔 접근

MinIO 콘솔은 내부망에 직접 공개하지 않습니다. Windows에서 SSH 터널을 연 상태로 사용합니다.

```powershell
ssh -L 9001:127.0.0.1:9001 ubuntu사용자명@192.168.0.3
```

브라우저에서 `http://localhost:9001`로 접속합니다. 터널을 닫으면 접근도 종료됩니다.

## 업데이트와 종료

Git에 새 커밋을 올린 뒤 VM에서 다음 명령으로 반영합니다.

```sh
git pull --ff-only
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
```

상태와 로그를 확인합니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs --tail=100
```

컨테이너만 내릴 때는 다음 명령을 사용합니다. 데이터 볼륨은 보존됩니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml down
```

`down -v`는 PostgreSQL과 MinIO 데이터도 삭제하므로 데이터 초기화가 명확히 필요한 경우에만
사용합니다. Windows PC 절전·종료 중에는 Ubuntu VM과 서비스도 사용할 수 없습니다.
