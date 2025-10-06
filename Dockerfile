


FROM continuumio/miniconda3:4.12.0

# FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*





WORKDIR /app

# copy yml
COPY environment.yml .


RUN conda env update -n base -f environment.yml && \
    conda clean -a -y


COPY . .


ENV PYTHONPATH=/app:${PYTHONPATH:-}

RUN cd /app/V2X-ATCT/MultiTest-master && \
    python build_script.py


CMD ["python", "Visualization/app.py"]
