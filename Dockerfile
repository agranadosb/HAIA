FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN mkdir /opt/HAIA
RUN mkdir /opt/HAIA/haia

ADD haia /opt/HAIA/haia
ADD __main__.py /opt/HAIA
ADD requirements.txt /opt/HAIA
ADD .env /opt/HAIA

WORKDIR /opt/HAIA

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/London apt-get -y install tzdata
RUN apt-get install -y nvidia-cuda-toolkit
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3.9-distutils -y
RUN python3.9 -m pip install --upgrade setuptools
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install --upgrade distlib
RUN python3.9 -m pip install torch==1.11.0+cu113 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.9 -m pip install -r /opt/HAIA/requirements.txt

ENTRYPOINT [ "/usr/bin/python3.9", "__main__.py" ]
