FROM --platform=${TARGETPLATFORM:-linux/amd64} 10.119.46.41:30003/cc/openfaas/of-watchdog:latest as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:3

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
RUN apt-get update \
    && apt-get install -y ca-certificates build-essential ${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .

COPY torch-1.12.1+cpu-cp310-cp310-linux_x86_64.whl .
COPY torchvision-0.13.1+cpu-cp310-cp310-linux_x86_64.whl .
RUN mkdir -p /home/app/python
ENV PATH=$PATH:/home/app/.local/bin:/home/app/python/bin/
ENV PYTHONPATH=$PYTHONPATH:/home/app/python
RUN pip install torch-1.12.1+cpu-cp310-cp310-linux_x86_64.whl
RUN pip install torchvision-0.13.1+cpu-cp310-cp310-linux_x86_64.whl
RUN rm torch-1.12.1+cpu-cp310-cp310-linux_x86_64.whl
RUN rm torchvision-0.13.1+cpu-cp310-cp310-linux_x86_64.whl
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
COPY model_server_for_cpu.py .
COPY init.sh .

# Allow any user-id for OpenShift users.


ENV fprocess="python3 index.py"
EXPOSE 8080

HEALTHCHECK --interval=10s CMD [ -e /tmp/.lock ] || exit 1

CMD ["bash","init.sh"]
