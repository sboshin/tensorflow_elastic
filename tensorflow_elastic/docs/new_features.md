# New feature designs

## Mid Epoch Recovery Design

* Save multi-device iterator
    * Implement Save
* Modify shard_num in shard_dataset.cc
* As long as next_index is the same as saved, but shard_num is modified to meet new cluster size, we shouldn’t repeat data
    * Test. list [11, 22, 333, 444, 555]
    * cluster size [2, 2, 3, 3, 3]
    * expected output for each node: 1, 2, 3, 4, 5
* New operator to Load Multi-device iterator with NewShard Number
    * Take old iterator, and load it into new dataset which has the correct shard number
* Add iterator recovery to Keras

## Worker Restart

The proposed Design will attempt to fix 2 things.

1. On shrink, old nodes should fix collective groups and keys
2. On Growth, new nodes should “warm up” and only join cluster when warm, and old nodes will fix collective groups and keys

### On Shrink

Hijack function defs in eager context, modify the collective group numbers, delete function cache and traces. Retrace
Problems

* Not sure if this will skip session create mechanism.
    * In theory, the executors and keys should remain the same, and the operator executors should already be created so it shouldn’t be a full recreation anyways.
* Will need to integrate tensorflow_elastic into a cluster resolver.
    * This will allow cluster spec to be dynamic.
* Verify variables are re-initialized from chief on reset

### On Growth

New nodes will be created on a completely separate cluster, and warmup. Warm up feature will have to be integrated into Keras, with a callback to the orchestrator that warmup is completed. Once warmup is completed initiate the “Shrink” procedure. Reset the whole cluster with the correct clusterspec and modify the collective keys.

## Recovery Mechanism through Orchestrator (Currently using cloud storage is sufficient)

Add API to the orchestrator to save models. If chief send saved artifacts to the orchestrator, else continue with normal backup procedure. Restore model will send saved artifacts from orchestrator to all nodes. (As all nodes have to restore from the same checkpoint)



