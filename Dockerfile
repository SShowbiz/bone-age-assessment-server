FROM python:3.7.2

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . /app

RUN dpkg --add-architecture i386; \
    apt-get update; \
    apt-get -y install libgl1-mesa-glx; \
    pip install --upgrade pip; \
    pip install -r requirements.txt

ENTRYPOINT ["python app.py"]