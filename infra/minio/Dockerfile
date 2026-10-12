# Community images are no longer distributed; build the official tagged source.
FROM golang:1.24-bookworm AS builder

ARG MINIO_VERSION=RELEASE.2025-10-15T17-29-55Z
WORKDIR /src
RUN git clone --depth 1 --branch "${MINIO_VERSION}" https://github.com/minio/minio.git .
RUN go mod download
ENV CGO_ENABLED=0 MINIO_RELEASE=RELEASE
# Limit compiler parallelism for Docker Desktop memory usage.
RUN go build -p 2 -trimpath -ldflags "$(go run buildscripts/gen-ldflags.go)" -o /out/minio .

FROM alpine:3.22
RUN apk add --no-cache ca-certificates curl \
    && addgroup -g 10001 minio \
    && adduser -D -u 10001 -G minio minio \
    && mkdir -p /data \
    && chown minio:minio /data
COPY --from=builder /out/minio /usr/local/bin/minio
COPY --from=builder /src/LICENSE /usr/share/licenses/minio/LICENSE
USER minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
