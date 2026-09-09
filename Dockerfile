FROM python:3.12-alpine AS builder
WORKDIR /project
COPY scripts/build_site.py scripts/build_site.py
COPY site site
COPY results results
COPY figures figures
COPY Group8-LLMOptimization.pdf Group8-LLMOptimization.pdf
RUN python3 scripts/build_site.py --output-dir /site

FROM nginxinc/nginx-unprivileged:1.31-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /site /usr/share/nginx/html
USER 101
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
