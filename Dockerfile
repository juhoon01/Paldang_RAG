# 파이썬 공식 이미지를 기반으로 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 파일 복사 및 설치
COPY app/requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN set -e; \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./

# 문서 데이터 복사
COPY data/ /app/data/

# FastAPI 앱 실행 명령 설정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]