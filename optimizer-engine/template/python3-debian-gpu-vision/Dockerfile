FROM --platform=${TARGETPLATFORM:-linux/amd64} 10.119.46.41:30003/cc/openfaas/of-watchdog:latest as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} docker.io/pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
RUN sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN  apt-get clean
RUN apt-get update \
    && apt-get install -y ca-certificates build-essential vim ffmpeg libsm6 libxext6 libgl1-mesa-glx libgl1-mesa-dev libglib2.0-dev libsndfile1 ${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .

RUN mkdir -p /home/app/python
ENV PATH=$PATH:/home/app/.local/bin:/home/app/python/bin/
ENV PYTHONPATH=$PYTHONPATH:/home/app/python
RUN pip install -r requirements.txt

RUN mkdir -p function
RUN touch ./function/__init__.py

WORKDIR /home/app/function/
COPY function/requirements.txt	.

RUN pip install -r requirements.txt

WORKDIR /home/app/

USER root

RUN chmod -R 777 /home/app/python

COPY function         function
COPY model_server_for_local.py .
COPY init.sh .

# Allow any user-id for OpenShift users.

ENV fprocess="python3 index.py"
EXPOSE 8080

HEALTHCHECK --interval=10s CMD [ -e /tmp/.lock ] || exit 1

CMD ["bash","init.sh"]
