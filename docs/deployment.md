# Linux 운영 배포 전환

현재 `compose.yaml`은 로컬 개발용입니다. Linux에서도 개발 환경으로 실행할 수 있지만,
운영 배포 및 HTTPS 설정은 아직 구현하지 않았습니다.

- 프런트엔드는 Node 빌드 단계에서 `npm ci && npm run build`를 실행하고,
  생성된 `dist`를 Nginx에서 정적으로 제공합니다. SPA 경로는 `try_files`로 처리합니다.
- 백엔드는 소스 bind mount와 `--reload`를 제거합니다.
- 도메인과 인증서를 준비해 Nginx에 HTTPS와 HTTP → HTTPS 리다이렉트를 설정합니다.
  `infra/nginx/certs`의 실제 인증서·개인 키는 Git에 올리지 않습니다.
- DB와 MinIO는 내부 네트워크에서만 접근하도록 호스트 포트 공개를 제거합니다.
  애플리케이션의 MinIO 접근에는 별도 제한된 서비스 계정을 사용합니다.
- 운영 환경변수, 데이터 백업, 재시작 정책, 비관리자 실행, 리소스 제한을 별도 배포 설정에 반영합니다.
- 현재 Python 의존성은 허용 범위로 선언되어 있습니다. 운영 도입 전 검증된 버전으로
  lockfile 또는 해시를 포함한 의존성 목록을 고정하고 컨테이너 이미지 digest도 고정합니다.
- MinIO는 공식 소스의 고정 릴리스 태그를 Docker에서 빌드합니다. 커뮤니티 저장소가
  보관 상태이므로 운영 도입 시 유지보수 가능한 배포판·S3 저장소를 확인합니다.
  [공식 저장소 및 사용한 릴리스](https://github.com/minio/minio/releases/tag/RELEASE.2025-10-15T17-29-55Z)

Airflow, MLflow, GPU/PyTorch 서비스는 필요 시 별도 Compose profile이나 파일로 추가합니다.
