# SmartNet Playground — CPU-only, optimized for t3a.micro (1GB RAM)
#
# Build:  docker build -t smartnet .
# Run:    docker run -p 5000:5000 smartnet
#
# Image:  ~1.2GB (torch CPU is large but runtime is lean)
# Memory: ~220MB runtime (fits t3a.micro 1GB easily)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only torch + flask, then strip unnecessary files
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask numpy && \
    # ---- Trim safely (saves ~100MB) ----
    SITE=/usr/local/lib/python3.11/site-packages && \
    rm -rf $SITE/torch/test \
           $SITE/torch/include \
           $SITE/torch/share && \
    find $SITE/sympy -name "tests" -type d -exec rm -rf {} + 2>/dev/null; \
    rm -rf $SITE/pip $SITE/setuptools /root/.cache /tmp/*

# Copy app files
COPY web_app/app.py          web_app/app.py
COPY web_app/__init__.py      web_app/__init__.py
COPY web_app/templates/       web_app/templates/
COPY web_app/static/css/      web_app/static/css/
COPY web_app/static/js/       web_app/static/js/

EXPOSE 5000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

CMD ["python", "-m", "web_app.app"]
