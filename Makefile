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
	docker run --gpus 1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v $(PWD)/model_repository:/models nvcr.io/nvidia/tritonserver:22.04-py3 \
		tritonserver --model-repository=/models

api:
	PYTHONPATH=src uvicorn api.main:app --reload --host 0.0.0.0 --port 8888

locust:
	locust -f src/locust/locustfile.py APIUser

# model
# you can use tensorrt for more optimized performance
# https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
train:
	echo "Training starts"
	PYTHONPATH=src/ml python src/ml/train.py
	echo "The trained model is save as model.pt"
	mkdir -p model_repository/mnist_cnn/1
	cp model.pt model_repository/mnist_cnn/1
	echo "model.pt is copied to model_repository/mnist_cnn/1"
