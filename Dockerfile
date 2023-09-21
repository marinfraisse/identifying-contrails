FROM python:3.10.6-buster as builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements_docker.txt /requirements.txt

RUN . /opt/venv/bin/activate
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY idcontrails/ idcontrails/
COPY setup.py setup.py

RUN pip install .



FROM python:3.10.6-slim-buster as runner

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder idcontrails/ idcontrails/

ENV PATH="/opt/venv/bin:$PATH"

CMD uvicorn idcontrails.interface.fast_test_api:app --host 0.0.0.0 --port $PORT
