import logging
import json
import grpc

import tensorflow_elastic.orchestrator_pb2 as orchestrator_pb2
import tensorflow_elastic.orchestrator_pb2_grpc as orchestrator_pb2_grpc



class TFEOrchestratorHandler():
  def __init__(self, server, min_nodes, max_nodes):
    self.server = server
    self.min_nodes = min_nodes
    self.max_nodes = max_nodes

  def GetClusterSpec(self, address, reset):
    with grpc.insecure_channel(self.server) as channel:
        stub = orchestrator_pb2_grpc.TFEOrchestratorStub(channel)
        ret = stub.GetClusterSpec(
          orchestrator_pb2.WorkerRegistration(address=address, min_nodes=self.min_nodes, max_nodes=self.max_nodes, reset=reset))
    return json.loads(ret.cluster_spec)

  def GetWaitingNodes(self, address):
    with grpc.insecure_channel(self.server) as channel:
        stub = orchestrator_pb2_grpc.TFEOrchestratorStub(channel)
        ret = stub.GetWaitingNodes(orchestrator_pb2.WaitNodesRequest(address=address))
    if(ret.error_msg != ""):
      raise ValueError(ret.error_msg)
    return ret.num_waiting_nodes

  def Barrier(self, address, tag, timeout):
    with grpc.insecure_channel(self.server) as channel:
        stub = orchestrator_pb2_grpc.TFEOrchestratorStub(channel)
        ret = stub.Barrier(orchestrator_pb2.BarrierRequest(address=address, tag=tag, timeout=timeout))
    if(ret.error_msg != ""):
      raise ValueError(ret.error_msg)
    return ret.success, ret.error_msg

  def Synchronize(self, address, tag, data, timeout, sleep=0):
    with grpc.insecure_channel(self.server) as channel:
        stub = orchestrator_pb2_grpc.TFEOrchestratorStub(channel)
        ret = stub.Synchronize(orchestrator_pb2.SyncRequest(address=address, tag=tag, data=data, timeout=timeout))
    if(ret.error_msg != ""):
      raise ValueError(ret.error_msg)
    return ret.success, json.loads(ret.data), ret.error_msg

  def ShutDown(self):
    with grpc.insecure_channel(self.server) as channel:
        stub = orchestrator_pb2_grpc.TFEOrchestratorStub(channel)
        ret = stub.ShutDown(orchestrator_pb2.ShutDownRequest())
    return ret.end_time
    