# 베이스 이미지 설정
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN apt-get update && \
    apt-get install -y cmake g++ make

# OpenGL 관련 패키지 설치
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libgthread-2.0-0

# 캐시 청소
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# 포트 설정
EXPOSE 5000

# 애플리케이션 실행
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
