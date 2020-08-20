#!/usr/bin/env python3

import logging
import os
import signal
import subprocess
import sys
import uuid
import socket
import multiprocessing as mp
from argparse import REMAINDER, ArgumentParser

from tensorflow_elastic import metrics
from tensorflow_elastic.agent.server.api import WorkerSpec
from tensorflow_elastic.agent.server.local_elastic_agent import LocalElasticAgent
from tensorflow_elastic.utils.logging import get_logger
from tensorflow_elastic.rendezvous import orchestrator_server
from tensorflow_elastic.rendezvous import orchestrator_api


log = get_logger()


def parse_args(args):
    """
    Helper function parsing the command line options.
    """

    parser = ArgumentParser(description="torchelastic elastic training launcher")

    # Arguments for the launch helper
    # worker/node size related arguments
    parser.add_argument(
        "--nnodes",
        type=str,
        default="1:1",
        help="number of nodes or MIN_NODES:MAX_NODES",
    )
    
    parser.add_argument(
        "--rdzv_endpoint",
        type=str,
        default="",
        help="rendezvous backend server host:port",
    )
    parser.add_argument("--rdzv_id", type=str, help="user defined group id")
    
    # sidecar embed rdzv backend that defults to etcd
    parser.add_argument(
        "--standalone",
        default=False,
        action="store_true",
        help="starts a local, standalone rdzv backend that is represented by"
        " etcd server on a random free port"
        "using the etcd binary specified in TORCHELASTIC_ETCD_BINARY_PATH"
        " env var or the one found in PATH."
        " Useful when launching single-node, multi-worker job."
        " If specified --rdzv_backend, --rdzv_endpoint, --rdzv_id"
        " are autoassigned, any explicitly set values are ignored",
    )

    # user-code launch related arguments
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=3,
        help="max number of worker group restarts before failing",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=5,
        help="interval (in seconds) to monitor the state of workers",
    )
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start_method to use when creating workers",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    parser.add_argument(
        "--address",
        default="",
        help="Local address and port eg: localhost:5000, if nothing is specified it will assign localhost + unused port"
        "If only port specified 5000, socket name + port used",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args(args)


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def PickUnusedPort():
  s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
  s.bind(('', 0))
  port = s.getsockname()[1]
  s.close()
  return port

def parse_address(arg_address):
  if(arg_address is ""):
    #Create an address and port
    return f"localhost:{PickUnusedPort()}"
  else:
    address_split = arg_address.split(":")
    if(len(address_split) < 2):
      #Only one thing specified
      if(address_split[0].isdigit()):
        #only digits its assumed a port else an ip
        return f"{socket.getfqdn(socket.gethostname())}:{address_split[0]}"
      else:
        return f"{address_split[0]}:{PickUnusedPort()}"
    else:
      return arg_address
      

def main(args=None):
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    args = parse_args(args)

    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0
    if args.standalone:
        endpoint_port = args.rdzv_endpoint.split(":")[-1] if args.rdzv_endpoint else None
        unused_port = endpoint_port or PickUnusedPort()
        #orchestrator = mp.Process(target=orchestrator_server.serve, args=(unused_port,))
        #orchestrator.start()
        args.rdzv_endpoint = f"localhost:{unused_port}"
        log.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv_endpoint={args.rdzv_endpoint} "
            f"Starting Server with at port {endpoint_port}"
            f"**************************************\n"
        )

        def kill_orchestrator(signum, frame):
          #orchestrator.terminate()
          #orchestrator.join()
          print(f"Signal {signum} was called, Exiting the Orchestrator")
          exit(0)

        signal.signal(signal.SIGTERM, kill_orchestrator)
        signal.signal(signal.SIGINT, kill_orchestrator)
        #while(true):
        #  orchestrator.join()
        orchestrator_server.serve(unused_port)

    with_python = not args.no_python
    cmd = []
    if with_python:
        cmd = [sys.executable, "-u"]
        if args.module:
            cmd.append("-m")
    else:
        if args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag"
                " and the '--module' flag at the same time."
            )

    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    rdzv_handler = orchestrator_api.TFEOrchestratorHandler(args.rdzv_endpoint, min_nodes, max_nodes)
    address = parse_address(args.address)

    try:
        spec = WorkerSpec(
            role="default",
            args=cmd,
            rdzv_handler=rdzv_handler,
            max_restarts=args.max_restarts,
            monitor_interval=args.monitor_interval,
            address=address,
        )
        metrics.initialize_metrics()
        elastic_agent = LocalElasticAgent(spec, start_method=args.start_method)
        elastic_agent.run(spec.role)
    finally:
        rdzv_handler.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
    )
    log.info(f"Running tensorflow_elastic.distributed.launch with args: {sys.argv}")
    main()
