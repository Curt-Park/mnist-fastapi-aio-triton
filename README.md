# NOT COMPLETLY DONE
You can see the previous works from:

1. https://github.com/Curt-Park/triton-inference-server-practice (00-quick-start)
2. https://github.com/Curt-Park/producer-consumer-fastapi-celery
3. https://github.com/Curt-Park/mnist-fastapi-celery-triton

# FastAPI + Triton (AsyncIO & gRPC)

## Preparation

#### 1. Setup packages
Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) and execute the following commands:

```bash
$ make env        # create a conda environment (need only once)
$ conda activate mnist-fastapi-aio-triton
$ make setup      # setup packages (need only once)
```

#### 2. Train a CNN model (Recommended on GPU)

```bash
$ make train
$ tree model_repository  # check the model repository created

model_repository
└── mnist_cnn
    ├── 1
    │   └── model.pt
    └── config.pbtxt

2 directories, 2 files
```

## How to play

#### Server

```bash
$ make triton     # run triton server
$ make api        # run fastapi server
```

- NOTE: If you want to run triton server and fastapi server on different devices, just set `TRITON_SERVER_URL` before running fastapi.
```bash
export TRITON_SERVER_URL=ip-address:8001
```

#### Execute Locust

```bash
$ make locust
```

Open http://0.0.0.0:8089 and type the api address in `Host`.
<img width="1275" alt="" src="https://user-images.githubusercontent.com/14961526/184040578-26b07242-c665-448e-9f5c-82988ffcc44b.png">

## Further Steps for k8s
- Triton + K8s: https://github.com/triton-inference-server/server/tree/main/deploy/k8s-onprem
- Locust + K8s: https://github.com/Curt-Park/locust-k8s
