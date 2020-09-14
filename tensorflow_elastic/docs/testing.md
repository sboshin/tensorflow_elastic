# Elastic Benchmarks

Description
The benchmarks are done by running tensorflow_elastic/tests/benchmark/run_aws_test.py

**_Ondemand Runs_**
--log_dir [logdir]  
--local   
--instance_type p3.16xlarge   
--instance_cnt [ondemand nodes]   
--ami [ami id]  
--key_name [key name]  
--setup_script ./test_setup_script.sh   

/home/ubuntu/benchmark/models/official/vision/image_classification/classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --data_dir=/home/ubuntu/imagenet/train/ --model_dir=/tmp/resnet --config_file=/home/ubuntu/benchmark/models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu_init_mwms.yaml -ara=nccl --params_override='train.epochs=20,model.learning_rate.warmup_epochs=5,model.learning_rate.multipliers="[.1, .01, .001, .001]",train.callbacks.backup_dir="s3://[s3bucket for backpu]"'

_**Elastic Runs:**_ *Add to ondemand runs*

--elastic_nodes [num elastic nodes]  
--elastic_pattern cycle:[cycle time]  

test_setup_script.sh

```
python -m pip install awscli -U
python -m pip install wrapt -U --ignore-installed
python -m pip install tensorflow==2.3.0 -U
python -m pip install [path to]/tensorflow_elastic-0.0.1-cp36-cp36m-linux_x86_64.whl -U
WORKING_DIR=/home/ubuntu/benchmark/models
if [ -d "$WORKING_DIR" ]; then rm -Rf $WORKING_DIR; fi
git clone -b elastic_bench https://github.com/sboshin/models.git $WORKING_DIR
python -m pip install -r /home/ubuntu/benchmark/models/official/requirements.txt
```

|Setup (Ondemand: Cycle)	|Epochs	|Time taken (seconds)	|Time taken minutes	|Accuracy	|top5	|Loss	|
|---	|---	|---	|---	|---	|---	|---	|
|4:4 (10 min cycle)	|	|5211.07127	|86.85119	|0.499	|0.7338	|4.8093	|
|4:0	|20	|2661.813	|44.36355	|0.5676	|0.7927	|3.282	|
|8:0	|20	|1648.06877	|27.46781	|0.4675	|0.706	|4.86	|
|4:4 (20 min cycle)	|20	|2772.40931	|46.20682	|0.4038	|0.6409	|4.098	|
|4:4 (15 min cycle)	|20	|3498.80775	|58.31346	|0.4789	|0.7175	|4.6105	|

<br>
<br>

## In-depth timing analysis (Timeline of events for each node)

|Epoch	|Ondemand 8	|Perf	|Elastic Cycle 20 min on and off 4 on demand 4 elastic	|Perf	|Difference (s)	|	|
|---	|---	|---	|---	|---	|---	|---	|
|	|	|	|	|	|	|	|
|	|Script setup took 53.775354623794556	|53.77535	|Script setup took 51.414146900177	|51.41415	|-2.36121	|	|
|0	|Epoch 0 took 375.15 loss: 11.93 accuracy: 0.0055 top_5_accuracy: 0.023	|375.15	|Epoch 0 took 360.79 loss: 12.0613 accuracy: 0.0044 top_5_accuracy: 0.0194	|360.79	|-14.36	|	|
|1	|Epoch 1 took 50.91 loss: 9.8364 accuracy: 0.0299 top_5_accuracy: 0.0986	|50.91	|Epoch 1 took 51.39 loss: 9.9856 accuracy: 0.0232 top_5_accuracy: 0.0798	|51.39	|0.48	|	|
|2	|Epoch 2 took 54.77 loss: 7.9667 accuracy: 0.0581 top_5_accuracy: 0.1641	|54.77	|Epoch 2 took 51.08 loss: 7.9713 accuracy: 0.045 top_5_accuracy: 0.1344	|51.08	|-3.69	|	|
|3	|Epoch 3 took 50.59 loss: 6.5985 accuracy: 0.0863 top_5_accuracy: 0.2212	|50.59	|Epoch 3 took 50.68 loss: 6.62 accuracy: 0.0727 top_5_accuracy: 0.1945	|50.68	|0.09	|	|
|4	|Epoch 4 took 51.15 loss: 6.1354 accuracy: 0.1123 top_5_accuracy: 0.2679	|51.15	|Epoch 4 took 52.11 loss: 6.2764 accuracy: 0.092 top_5_accuracy: 0.2295	|52.11	|0.96	|	|
|5	|Epoch 5 took 51.97 loss: 5.7222 accuracy: 0.181 top_5_accuracy: 0.3726	|51.97	|Epoch 5 took 51.59 loss: 6.0356 accuracy: 0.1342 top_5_accuracy: 0.2964	|51.59	|-0.38	|	|
|6	|Epoch 6 took 52.08 loss: 5.0753 accuracy: 0.2624 top_5_accuracy: 0.4872	|52.08	|Epoch 6 took 51.53 loss: 5.3155 accuracy: 0.2167 top_5_accuracy: 0.4264	|51.53	|-0.55	|	|
|7	|Epoch 7 took 50.46 loss: 4.752 accuracy: 0.3012 top_5_accuracy: 0.5351	|50.46	|Epoch 7 took 51.85 loss: 4.9769 accuracy: 0.2564 top_5_accuracy: 0.4791	|51.85	|1.39	|	|
|8	|Epoch 8 took 50.35 loss: 4.5036 accuracy: 0.3334 top_5_accuracy: 0.5723	|50.35	|Epoch 8 took 49.81 loss: 4.7228 accuracy: 0.2884 top_5_accuracy: 0.519	|49.81	|-0.54	|	|
|9	|Epoch 9 took 51.05 loss: 4.3062 accuracy: 0.3616 top_5_accuracy: 0.6038	|51.05	|Epoch 9 took 49.50 loss: 4.5168 accuracy: 0.3174 top_5_accuracy: 0.5543	|49.5	|-1.55	|	|
|10	|Epoch 10 took 50.78 loss: 4.1553 accuracy: 0.3863 top_5_accuracy: 0.6292	|50.78	|Epoch 10 took 50.54 loss: 4.3503 accuracy: 0.3437 top_5_accuracy: 0.5841	|50.54	|-0.24	|	|
|11	|Epoch 11 took 50.09 loss: 4.0281 accuracy: 0.4093 top_5_accuracy: 0.6522	|50.09	|Epoch 11 took 49.52 loss: 4.2101 accuracy: 0.3689 top_5_accuracy: 0.6102	|49.52	|-0.57	|	|
|	|	|	|initialize_workers took 33.626	|	|0	|	|
|	|	|	|Script setup took 27.135733366012573	|	|0	|	|
|	|	|	|Epoch time lost 115.321697473526	|115.3217	|	|	|
|12	|Epoch 12 took 50.24 loss: 3.9328 accuracy: 0.4276 top_5_accuracy: 0.6706	|50.24	|Epoch 12 took 675.68 loss: 4.0457 accuracy: 0.3986 top_5_accuracy: 0.6414	|675.68	|625.44	|	|
|13	|Epoch 13 took 50.77 loss: 3.8558 accuracy: 0.4449 top_5_accuracy: 0.6871	|50.77	|Epoch 13 took 79.69 loss: 3.9269 accuracy: 0.4199 top_5_accuracy: 0.6629	|79.69	|28.92	|	|
|14	|Epoch 14 took 51.02 loss: 3.8352 accuracy: 0.4546 top_5_accuracy: 0.6961	|51.02	|Epoch 14 took 78.58 loss: 3.8304 accuracy: 0.4382 top_5_accuracy: 0.68	|78.58	|27.56	|	|
|15	|Epoch 15 took 51.06 loss: 3.8352 accuracy: 0.4679 top_5_accuracy: 0.7082	|51.06	|Epoch 15 took 77.19 loss: 3.7505 accuracy: 0.4539 top_5_accuracy: 0.6958	|77.19	|26.13	|	|
|16	|Epoch 16 took 50.70 loss: 6.1357 accuracy: 0.2606 top_5_accuracy: 0.4649	|50.7	|Epoch 16 took 75.00 loss: 3.6842 accuracy: 0.4678 top_5_accuracy: 0.7086	|75	|24.3	|	|
|17	|Epoch 17 took 50.22 loss: 5.5254 accuracy: 0.4011 top_5_accuracy: 0.6418	|50.22	|Epoch 17 took 74.13 loss: 3.625 accuracy: 0.4818 top_5_accuracy: 0.7206	|74.13	|23.91	|	|
|18	|Epoch 18 took 50.31 loss: 5.0773 accuracy: 0.449 top_5_accuracy: 0.6887	|50.31	|Epoch 18 took 74.05 loss: 3.5805 accuracy: 0.4915 top_5_accuracy: 0.7295	|74.05	|23.74	|	|
|	|	|	|initialize_workers took 3.009	|	|0	|	|
|	|	|	|Script setup took 28.11906361579895	|	|0	|	|
|	|	|	|Epoch time lost 84.82868695259094	|84.82869	|	|	|
|19	|Epoch 19 took 50.23 loss: 4.8555 accuracy: 0.4675 top_5_accuracy: 0.706	|50.23	|Epoch 19 took 448.04 loss: 4.0984 accuracy: 0.4038 top_5_accuracy: 0.6409	|448.04	|397.81	|	|
|20	|Total train time is 1648.0687718391418	|1648.06877	|Total train time is 2772.4093136787415	|2772.40931	|1124.34054	|	|


