FROM python:3.10-slim
WORKDIR /runpod-volume
COPY . .
RUN pip install -r requirements.txt
CMD ["python3", "main.py"]