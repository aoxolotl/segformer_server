FROM python:3.9

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6' -y

RUN apt install -y liblzma-dev

WORKDIR /workspace
COPY requirements.txt /workspace
RUN pip install -r requirements.txt

# Download model to local cache
COPY download_to_cache.py /workspace
ENV TRANSFORMERS_CACHE=./cache/
RUN python download_to_cache.py

# install gunicorn for wsgi
RUN pip install gunicorn==20.0.4
COPY server.py /workspace

EXPOSE 8000
ENV DEVICE=cpu
WORKDIR /workspace
CMD [ "gunicorn", "-w 6", "-b 0.0.0.0:8000", "server:app" ]
