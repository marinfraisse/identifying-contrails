FROM python:3.10.6-buster
COPY requirements_docker.txt /requirements_docker.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt

COPY idcontrails / idcontrails
#à la fin car sionn ca relaod pip install (docker check les modifs de haut en bas)

CMD uvicorn idcontrails.interface.fast_test_api:app --host 0.0.0.0
