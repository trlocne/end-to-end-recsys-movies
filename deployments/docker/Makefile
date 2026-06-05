# Build / push images from repo root as context.
#   make -C deployments/docker build-recsys-ui REGISTRY=your-dockerhub-user
#   docker login … && make -C deployments/docker push-recsys-ui REGISTRY=your-dockerhub-user
#
ROOT := $(abspath $(CURDIR)/../..)
DOCKERFILE_UI := $(CURDIR)/services/Dockerfile.recsys-ui
DOCKERFILE_SERVING := $(CURDIR)/services/Dockerfile.serving
REGISTRY ?= trlocne204
TAG ?= latest
IMAGE_UI := $(REGISTRY)/recsys-ui:$(TAG)
IMAGE_SERVING := $(REGISTRY)/serving:$(TAG)

.PHONY: build-recsys-ui push-recsys-ui build-serving push-serving

build-recsys-ui:
	docker build -f "$(DOCKERFILE_UI)" -t "$(IMAGE_UI)" "$(ROOT)"

push-recsys-ui: build-recsys-ui
	docker push "$(IMAGE_UI)"

build-serving:
	docker build -f "$(DOCKERFILE_SERVING)" -t "$(IMAGE_SERVING)" "$(ROOT)"

push-serving: build-serving
	docker push "$(IMAGE_SERVING)"
