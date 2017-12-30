FROM python:3

ADD . /src
RUN pip install -r /src/requirements.txt

CMD ["python", "/src/dqn.py"]
