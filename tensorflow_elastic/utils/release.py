#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
import tensorflow_elastic.rendezvous.orchestrator_api as orch_api

def start_health_check(orchestrator_ip, interval, params):
  minN = params["minN"]
  maxN = params["maxN"]
  handler = orch_api.TFEOrchestratorHandler(orchestrator_ip, minN, maxN)
  def _start():
    time.sleep(interval)
    handler.GetWaitingNodes(orchestrator_ip)

  t = threading.Thread(target=_start)
  return t