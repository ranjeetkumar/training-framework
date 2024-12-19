FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
COPY framework/ /app/framework/
COPY example_training_script.py /app/

RUN pip install tensorflow

