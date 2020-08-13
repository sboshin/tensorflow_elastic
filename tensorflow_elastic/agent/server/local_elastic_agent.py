#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from typing import Any, Dict

#import torch.multiprocessing as mp
from tensorflow_elastic.agent.server.api import (
    MonitorResult,
    SimpleElasticAgent,
    Worker,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
import multiprocessing as mp
from tensorflow_elastic.metrics.api import prof
from tensorflow_elastic.utils.logging import get_logger


log = get_logger()


class _DistInfo:
    """
    Container for information required to create a torch process group.
    To be created on the agent's process and passed to the worker sub-process.
    Hence this object needs to be a pure data object with no state and
    preferably only primitive member variables
    """

    __slots__ = [
        "restart_count",
        "max_restarts",
        "cluster_spec",
        "group_rank"
    ]

    def __init__(
        self,
        restart_count: int,
        max_restarts: int,
        cluster_spec: str,
        group_rank: int,
    ):
        self.restart_count = restart_count
        self.max_restarts = max_restarts
        self.cluster_spec = cluster_spec
        self.group_rank = group_rank


def _wrap(local_rank, ret_vals, dist_infos, fn, args):
    import faulthandler

    try:
        faulthandler.enable(all_threads=True)
        pass
    except Exception as e:
        log.warn(
            "Unable to enable fault handler. Failure signals on worker process will not dump tracebacks",
            exc_info=e,
        )

    info = dist_infos[local_rank]
    os.environ["TORCHELASTIC_RESTART_COUNT"] = str(info.restart_count)
    os.environ["TORCHELASTIC_MAX_RESTARTS"] = str(info.max_restarts)
    os.environ["RANK"] = str(info.group_rank)
    #We are faking the cluster at this point
    print(dist_infos[local_rank].cluster_spec)
    
    tf_config = dist_infos[local_rank].cluster_spec
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    ret = fn(*args)
    ret_vals[info.group_rank] = ret 


class ProcessContext(object):

  def __init__(self, procs):
    self.processes = procs
    
  def pids(self):
    return [proc.pid for proc in self.processes]

  def join(self, timeout=-1):
    exit_codes = {proc.pid:proc.exitcode for proc in self.processes}
    for pid in exit_codes:
      code = exit_codes[pid]
      if code is not None and code is not 0:
        raise ChildProcessError(f"Process {pid} Exited with code {code}")
      elif code is None:
        return False
    return True



class LocalElasticAgent(SimpleElasticAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers.
    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.

    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The exit_barrier_timeout specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    Example

    ::

        def trainer(shared_queue):
            pass

        def main():
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        fn=trainer,
                        args=(shared_queue,),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            agent.run()
    """

    def __init__(
        self, spec: WorkerSpec, start_method="spawn", exit_barrier_timeout: float = 300
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        # pyre-fixme[8]: Attribute has type `ProcessContext`; used as `None`.
        self._process_context: ProcessContext = None
        # a map that holds return values for each worker fn
        # ret_val[0] holds the return value for worker_0 (global rank 0)
        self._manager = mp.get_context(start_method).Manager()
        self._ret_vals = self._manager.dict()

    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        for proc in self._process_context.processes:
            if proc.is_alive():
                proc.terminate()
                #proc.kill()
            proc.join()

    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        restart_count = spec.max_restarts - self._remaining_restarts

        dist_infos: Dict[int, _DistInfo] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            dist_infos[local_rank] = _DistInfo(
                restart_count,
                spec.max_restarts,
                self.cluster_spec,
                worker_group.group_rank,
            )

        self._ret_vals.clear()
        proc = mp.Process(
            target=_wrap,
            args=(0, self._ret_vals, dist_infos, spec.fn, spec.args),
            #nprocs=spec.local_world_size,
            #join=False,
            #daemon=False,
            #start_method=self._start_method,
        )
        proc.start()
        self._process_context = ProcessContext([proc])

        return {
            local_rank: pid
            for local_rank, pid in enumerate(self._process_context.pids())
        }

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> MonitorResult:
        role = worker_group.spec.role

        # torch process context join() isn't really a join in the
        # traditional sense, it returns True if all the workers have
        # successfully finished, False if some/all are still running
        # and throws an Exception if some/all of them failed
        # passing timeout < 0 means check worker status and return immediately

        worker_pids = {w.id for w in worker_group.workers}
        pc_pids = set(self._process_context.pids())
        if worker_pids != pc_pids:
            log.error(f"[{role}] worker pids do not match process_context pids")
            return MonitorResult(WorkerState.UNKNOWN)

        try:
            if self._process_context.join(timeout=-1):
                # copy ret_vals since we do not want to return an mp map
                return MonitorResult(WorkerState.SUCCEEDED, dict(self._ret_vals))
            else:
                return MonitorResult(WorkerState.HEALTHY)
        except Exception as e:
            log.exception(f"[{role}] Worker group failed")
            return MonitorResult(
                WorkerState.FAILED,
                exceptions={w.global_rank: e for w in worker_group.workers},
            )
