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

Open http://0.0.0.0:8089
