PYTHON=3.9
CONDA_CH=defaults conda-forge pytorch
BASENAME=$(shell basename $(CURDIR))

# setup
env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install -y --file requirements.txt $(addprefix -c ,$(CONDA_CH))
	pip install -r requirements-pip.txt  # separated for M1 chips


# services
triton:
	docker run --gpus 1 --ipc host --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v $(PWD)/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 \
		tritonserver --model-repository=/models

api:
	PYTHONPATH=src uvicorn api.main:app --reload --host 0.0.0.0 --port 8888
