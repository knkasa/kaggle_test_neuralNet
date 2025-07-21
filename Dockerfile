FROM public.ecr.aws/docker/library/python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "test2_autogluon_knn_aws.py"]
