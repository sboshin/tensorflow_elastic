# Tensorflow Elastic
This repository is for users who want to run the tensorflow multi-mirrored worker strategy, while being able to utilize pre-emptive instances, and dynamically allocated resources to training jobs. 

## Introduction

Tensorflow Elastic aims to provide APIs to allow users to utlize resources dynamically while training. Currently when users train, resources are statically allocated. For situations where resources are free-ed up and can be allocated to a job thats already running, Tensorflow Elastic allows users to capitalize on these situations. Also several cloud providers have certain instances which are available at steep discounts, but are pre-emptive, Tensorflow Elastic aims to also capitalize on these instances.

## Installation

Included in the repository is the install.sh bash script
```
chmod +x install.sh
./install.sh

or

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install -U artifacts/tensorflow_elastic-0.0.1-cp36-cp36m-linux_x86_64.whl
```

## High Level Overview [This will change frequently]

The current Tensorflow Elastic Design uses an orchestrator to organize the training cluster. Workers that want to join will register with the orchestrator, and the cluster will reset to accomodate the new nodes. Also on node failures, the orchestrator will have a lease for each worker, if workers don't renew, they are suspected to be down. This will trigger a rediscovery of the training cluster.

Planned Improvement: 
* Integrate Orchestrator into a strategy
* Integrate restarting mechanism into eager/graph context

Hopeful Improvements:
* Mid Epoch recovery
* Orchestrator recovery

## Setup Instructions

Install tensorflow_elastic on all instances including orchestrator

The Orchestrator: Started on a non preemptible system, this system can be a smaller instances as the orchestrator doesn't use much compute resources

Open ports on each instance, Each worker should be able to access every other worker, and the orchestrator.

## Usage Guide

Start orchestrator
```
python -m tensorflow_elastic.distributed.launch --standalone --rdzv_endpoint=localhost:[open port] start
```
Setup worker nodes as you would for multi-worker mirrored strategy

Start Worker nodes
```
python3 -m tensorflow_elastic.distributed.launch --monitor_interval=[integer in seconds] --address=[current worker's address :port] --nnodes=min_nodes:max_nodes --rdzv_endpoint=orchestrator ip:port multi_worker_script multi_worker_args
```
**Arguments**

* Monitor_interval: How often each worker renews their lease, and checks on the health of the running script
* address: This is the address of the worker, This is a public ip address and port thats open to allow the multi-worker mirrored strategy to work 
* nnodes: mininum number of nodes to start training : maximum number of nodes in the cluster
* rdzv_endpoint: The address and port of the orchestrator
* Rest will be considered as the multi worker script and its arguments

## End to End Examples
```
git clone https://github.com/sboshin/tensorflow_elastic.git
#Start orchestrator
python -m tensorflow_elastic.distributed.launch --standalone --rdzv_endpoint=localhost:5556 start

#Start two workers
python -m tensorflow_elastic.distributed.launch --monitor_interval=5 --address=localhost:5555 --nnodes=1:2 --rdzv_endpoint=localhost:5556 tensorflow_elastic/tensorflow_elastic/tests/bin/mnist_mwms.py &

#You can wait for 35 seconds between invocations
sleep 35

python -m tensorflow_elastic.distributed.launch --monitor_interval=5 --address=localhost:5557 --nnodes=1:2 --rdzv_endpoint=localhost:5556 tensorflow_elastic/tensorflow_elastic/tests/bin/mnist_mwms.py &
```


## Running Unit tests
```
bazel test //tensorflow_elastic/...
```

## Contributing
We welcome community contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) and, for style help,
[Writing TensorFlow documentation](https://www.tensorflow.org/community/documentation)
guide.

## License
[Apache 2.0 License](LICENSE)

