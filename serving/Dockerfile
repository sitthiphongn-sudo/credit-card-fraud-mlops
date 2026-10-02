FROM python:3.12-slim

WORKDIR /app

# libgomp1: LightGBM wheel ต้องใช้ OpenMP runtime ซึ่ง python:slim ไม่มีมาให้
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PYTHONPATH มี /app/src เพราะโมเดลที่ pickle ไว้อ้างคลาส fraud.features.FraudFeatureTransformer
# PROMETHEUS_MULTIPROC_DIR ทำให้ /metrics รวมค่าจากทุก worker ได้ถูกต้องเมื่อ WEB_CONCURRENCY > 1
# WEB_CONCURRENCY = จำนวน uvicorn worker (uvicorn อ่านค่านี้เอง) ค่าเริ่มต้น 1 ปรับได้ตอน run ตามผล load test
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc \
    WEB_CONCURRENCY=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY serving/ ./serving/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# ล้างไฟล์ metric ของรอบก่อน (กรณี container restart) แล้วเริ่ม uvicorn
CMD ["sh", "-c", "rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\" && exec uvicorn serving.app:app --host 0.0.0.0 --port 8000"]
