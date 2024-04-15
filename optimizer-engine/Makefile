.PHONY: build push redeploy
IMG_NAME?=smiless-optimizer-engine

TAG?=latest
SERVER?=blinkbear

all: build-docker

build-docker:
	docker build -t $(SERVER)/$(IMG_NAME):$(TAG) . && \
	docker push $(SERVER)/$(IMG_NAME):$(TAG) && \
	kubectl apply -f k8s_yaml/optimizer-engine.yaml

build:
	docker build -t $(SERVER)/$(IMG_NAME):$(TAG) .

push:
	docker push $(SERVER)/$(IMG_NAME):$(TAG)

redeploy:
	kubectl delete -f k8s_yaml/optimizer-engine.yaml && kubectl apply -f k8s_yaml/optimizer-engine.yaml