FROM python:3.8

ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code

COPY requirements.txt /code/
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /code/
CMD ["gunicorn", "--bind", ":8050", "--workers", "1", "--threads", "8", "--timeout", "0", "Dash_app:server"]
