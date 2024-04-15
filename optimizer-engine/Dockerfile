FROM python:3.9

WORKDIR /home/smiless
# copy all files to the container
RUN apt update && apt install nfs-common -y
COPY requirements.txt ./
# Install dependencies with requirements.txt

RUN pip install -r requirements.txt

COPY . .
RUN rm -r ./template
CMD ["python", "main.py"]