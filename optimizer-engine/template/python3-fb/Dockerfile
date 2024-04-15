FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/classic-watchdog:0.2.1 as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
RUN apt-get update && \
    apt-get install -y ca-certificates ${ADDITIONAL_PACKAGE} python3-pip && \
    rm -rf /var/lib/apt/lists/

WORKDIR /home/app/
COPY index.py           .
COPY requirements.txt   .
COPY torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl .
COPY torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl .
RUN  mkdir -p /home/app/python

ENV PATH=$PATH:/home/app/.local/bin:/home/app/python/bin/
ENV PYTHONPATH=$PYTHONPATH:/home/app/python
RUN pip3 install -r requirements.txt --target=/home/app/python
RUN pip3 install torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl
RUN pip3 install torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl


RUN rm torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl
RUN rm torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl
RUN mkdir -p function

RUN touch ./function/__init__.py

WORKDIR /home/app/function/
COPY function/requirements.txt	.

RUN pip3 install -r requirements.txt --target=/home/app/python
USER root
RUN chown -R 777 /home/app/python

WORKDIR /home/app/

USER root
COPY models             models
COPY init.sh .
COPY fb_server.py .
COPY function           function

ENV fprocess="python3 index.py"
EXPOSE 8080

HEALTHCHECK --interval=10s CMD [ -e /tmp/.lock ] || exit 1

ENTRYPOINT ["bash","init.sh"]
