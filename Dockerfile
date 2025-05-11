FROM python:3.10-slim
WORKDIR /runpod-volume
COPY . .
RUN pip install -r requirements.txt && python3 -m spacy download en_core_web_lg
CMD ["python3", "main.py"]