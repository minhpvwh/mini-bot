FROM tiangolo/meinheld-gunicorn-flask:python3.7
LABEL pyemteller.version="0.1" pyemteller.release-date="2020-05-25"

WORKDIR /app

COPY requirements.txt .
RUN  pip install -r ./requirements.txt

COPY . ./
ENTRYPOINT [ "python3" ]
CMD [ "api.py" ]
