FROM python:3.10

COPY requirements.txt /opt/ml/requirements.txt

RUN pip3 install --no-cache-dir -r /opt/ml/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"

COPY code /opt/ml/code
WORKDIR /opt/ml/code

ENV SAGEMAKER_PROGRAM train.py
