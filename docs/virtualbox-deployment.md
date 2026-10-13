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
MINIO_API_BIND_ADDRESS=192.168.0.3
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
접속할 수 있지만 공유기의 포트 포워딩으로 외부에 공개하지 않습니다. MinIO S3 API `9000`은 VM의 내부망 IP에
바인딩해 개발 PC에서 직접 사용합니다. 관리 콘솔 `9001`은 localhost 전용으로 두고 SSH 터널로 접근합니다.
S3 API는 현재 HTTP이므로 내부망의 다른 기기에서 네트워크를 관찰할 수 있으면 요청과 자격 증명이 노출될 수 있습니다.
신뢰하는 내부망에서만 사용하고, 개발 PC에는 root 계정 대신 모델 버킷만 허용된 개발 계정을 설정합니다.

개발 PC용 MinIO 계정과 전용 버킷은 아래 “개발 PC에서 서버 MinIO 사용” 절차로 초기화합니다. 서버 앱의
`S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY` 값은 비워 두어 서버의 기존 MinIO 계정 사용을 유지합니다.

MinIO API 포트를 내부망에 바인딩하기 전에 Docker `DOCKER-USER` 체인에서 내부망만 허용합니다. 예시는
서버/개발 PC가 `192.168.0.0/24` 대역일 때이며, 다른 대역이면 CIDR과 서버 IP를 조정합니다.

```sh
sudo iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctorigdst 192.168.0.3 --ctorigdstport 9000 -s 192.168.0.0/24 -j ACCEPT
sudo iptables -I DOCKER-USER 2 -p tcp -m conntrack --ctorigdst 192.168.0.3 --ctorigdstport 9000 -j DROP
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

## 실행과 확인

서버용 override는 VM 재부팅 후 컨테이너가 다시 시작되도록 정책을 추가합니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
```

MinIO 컴파일과 Airflow·PyTorch 학습 이미지의 패키지 다운로드 때문에 최초 빌드에는 시간이 걸리고
여러 GB의 디스크 공간을 사용합니다.

`mlflow` 로그에 `database "mlflow" does not exist`가 나오면 최신 `compose.yaml`을 pull한 뒤
아래 `up`을 다시 실행합니다. `tracking-db-init`이 기존 PostgreSQL 볼륨 안에 `airflow`와 `mlflow`
데이터베이스를 생성합니다. 애플리케이션 DB나 볼륨을 삭제할 필요가 없습니다.

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
경우 실제 내부망 CIDR만 허용합니다. 예시는 내부망이 `192.168.0.0/24`일 때입니다. Docker의 게시 포트는
UFW 규칙을 우회할 수 있으므로 MinIO API는 `DOCKER-USER` 체인에서도 내부망만 허용합니다.

```sh
sudo ufw allow OpenSSH
sudo ufw allow from 192.168.0.0/24 to any port 8080 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 5432 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 8081 proto tcp
sudo ufw allow from 192.168.0.0/24 to any port 5000 proto tcp
sudo ufw enable
```

Docker가 게시한 MinIO 포트는 UFW 규칙만으로 차단되지 않을 수 있습니다. `DOCKER-USER` 규칙이 설정되어 있는지
`sudo iptables -S DOCKER-USER`로 확인하고, 변경하거나 다시 추가할 때는 위 MinIO 내부망 허용 절차를 따르세요.

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

## 개발 PC에서 서버 MinIO 사용

S3 API는 `http://192.168.0.3:9000`으로 직접 연결합니다. MinIO 콘솔 `9001`은 계속 localhost 전용이므로
콘솔 관리가 필요할 때만 아래 SSH 터널을 엽니다. 모델 업로드·다운로드에는 SSH 터널이 필요하지 않습니다.

Ubuntu 프로젝트 폴더에서 서버 전용 개발 계정을 생성합니다. `.env.server`의 `DEV_S3_ACCESS_KEY_ID`와
`DEV_S3_SECRET_ACCESS_KEY`를 먼저 실제 값으로 설정하세요. `.env.server`가 이미 존재한다면 Git pull로 새 설정이
추가되지 않으니 직접 편집합니다. 예를 들어 아래 access key는 예제 값으로 쓸 수 있고, secret은 별도로 생성합니다.

또한 기존 `.env.server`에는 `MINIO_API_BIND_ADDRESS=192.168.0.3`을 직접 추가합니다. 앞서 설명한 `DOCKER-USER`
허용 규칙이 적용되어 있는지 확인한 다음 이 설정을 적용하도록 MinIO 컨테이너를 재생성하세요.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up -d --force-recreate minio
```

Windows에서 포트 연결을 확인합니다.

```powershell
Test-NetConnection 192.168.0.3 -Port 9000
```

Mac에서는 `nc -vz 192.168.0.3 9000`으로 확인합니다.

```sh
openssl rand -hex 32
```

`.env.server`에 다음 줄을 추가하고 secret 자리에 생성한 값을 입력합니다.

```dotenv
DEV_S3_ACCESS_KEY_ID=spot_fit_dev
DEV_S3_SECRET_ACCESS_KEY=생성한_랜덤_값
```

초기화 명령은 `spot-fit-models` 버킷을 준비하고, 그 버킷에만 읽기·쓰기 권한이 있는 개발 계정을 MinIO에
등록합니다. 다시 실행해도 `.env.server`의 같은 자격 증명이 유지됩니다.

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml --profile admin run --build --rm minio-dev-bootstrap
```

개발 PC 프로젝트 루트의 `.env`에 아래 값만 설정합니다. 기존 `.env`에도 새 항목은 자동 추가되지 않으므로 직접
추가합니다. 서버 `.env.server` 전체를 복사하지 않습니다.

```dotenv
DEV_S3_ENDPOINT_URL=http://192.168.0.3:9000
DEV_S3_MODEL_BUCKET=spot-fit-models
DEV_S3_ACCESS_KEY_ID=spot_fit_dev
DEV_S3_SECRET_ACCESS_KEY=Ubuntu의_DEV_S3_SECRET_ACCESS_KEY와_같은_값
```

변경 파일을 GitHub에 올린 뒤 개발 PC에서 pull하고, 위 접속 테스트가 성공하는지 확인합니다. 그 다음 `compose.minio-server.yaml`을
추가로 지정한 `model-tools` 명령은 로컬 MinIO 의존성을 시작하지 않고 Ubuntu 서버에 접속합니다.

```sh
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools build model-tools
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools init
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools upload --file /artifacts/models/recommender.joblib --name recommender --version v1
```

서버에 올린 모델 파일은 공유 버킷에 저장되고 manifest는 개발 PC의 `model_registry/`에 생성됩니다. manifest는 Git에
커밋해 공유하고 모델 파일은 Git에 커밋하지 않습니다. 내려받기도 동일한 두 Compose 파일과 `--no-deps`를 사용합니다.

```sh
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools download --manifest /model_registry/recommender/v1.json
```

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
