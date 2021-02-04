FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
ENV GOOGLE_APPLICATION_CREDENTIALS fyp-backend-owner-sa.json
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
COPY . .
