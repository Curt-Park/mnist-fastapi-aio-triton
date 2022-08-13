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

## Experimental Result
- CPU: AMD Ryzen Threadripper PRO 3995WX 64-Cores
- GPU: NVIDIA GeForce RTX 3090
- FastAPI, Triton, Locust are executed on the same device
- a single v-user sends a request once a second

<img width="1272" alt="" src="https://user-images.githubusercontent.com/14961526/184470769-cddcce5d-a394-45de-b94d-d92326eebae1.png">
<img width="1255" alt="" src="https://user-images.githubusercontent.com/14961526/184470771-17d4db16-f5c5-4cad-982f-854f385616c9.png">

From 1,400 v-user, the response latency increases.

The following is the triton metrics.
The difference of `nv_inference_count` and `nv_inference_exec_count` shows the dynamic batching works.
(Details about [Triton Metrics](https://github.com/triton-inference-server/server/blob/main/docs/metrics.md))

```bash
# Triton Metrics
$ curl localhost:8002/metrics

# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="mnist_cnn",version="1"} 395927.000000
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="mnist_cnn",version="1"} 0.000000
# HELP nv_inference_count Number of inferences performed (does not include cached requests)
# TYPE nv_inference_count counter
nv_inference_count{model="mnist_cnn",version="1"} 395927.000000
# HELP nv_inference_exec_count Number of model executions performed (does not include cached requests)
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="mnist_cnn",version="1"} 193751.000000
...
```

* NOTE: The connection number is limited to `open files` (check this with `ulimit -a` command).
In order to change the upper bound, you should set the maximum number of `open files` by `ulimit -Sn 65535`.

```bash
$ ulimit -a

core file size          (blocks, -c) 0
data seg size           (kbytes, -d) unlimited
scheduling priority             (-e) 0
file size               (blocks, -f) unlimited
pending signals                 (-i) 513958
max locked memory       (kbytes, -l) unlimited
max memory size         (kbytes, -m) unlimited
open files                      (-n) 65535    # <- This line!
pipe size            (512 bytes, -p) 8
POSIX message queues     (bytes, -q) 819200
real-time priority              (-r) 0
stack size              (kbytes, -s) 8192
cpu time               (seconds, -t) unlimited
max user processes              (-u) 513958
virtual memory          (kbytes, -v) unlimited
file locks                      (-x) unlimited
```

## Further Steps for k8s
- Triton + K8s: https://github.com/triton-inference-server/server/tree/main/deploy/k8s-onprem
- Locust + K8s: https://github.com/Curt-Park/locust-k8s
