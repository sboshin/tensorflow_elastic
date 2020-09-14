import os
import sys
import re
from argparse import ArgumentParser


def parse_args(args):
  parser = ArgumentParser(description="MWMS aws runner")
  parser.add_argument(
    "--log_dir", type=str, help="Directory with logs"
  )
  
  parser.add_argument(
    "--log_file", type=str, help="Log file"
  )

  parser.add_argument(
    "--output_timestamp", action="store_false", help="output timestamp for different events"
  )

  return parser.parse_args(args)

def match_timestamp_lines(line):
  #Match all generic lines
  patterns = []
  patterns.append("INFO:absl:Script starts at (.+)")
  patterns.append(".+keras_utils.py.+?Training begins at (.+)")
  patterns.append(".+keras_utils.py.+?Finished Training at ([0-9\.]+)")

  for each in patterns:
    ro = re.match(each, line)
    if ro is not None:
      return float(ro.group(1))
  
  return None


def match_epoch_lines(line):
  pattern = [".+keras_utils.py.+?Epoch (\d+) begins at (.+)", 
             ".+keras_utils.py.+?Epoch (\d+) ends at ([0-9\.]+), duration of (.+)"]

  for each in pattern:
    ro = re.match(each, line)
    if ro is not None:
      return (int(ro.group(1)), float(ro.group(2)))

  return None

def match_elatic_metrics(line):
  patterns = [".+SimpleElasticAgent\._initialize_workers\.duration\.ms=(\d+)"]

  for each in patterns:
    ro = re.match(each, line)
    if ro is not None:
      return int(ro.group(1))/1000

  return None

def match_loss_metrics(line):
  patterns = [".+loss: ([0-9\.]+) - accuracy: ([0-9\.]+) - top_5_accuracy: ([0-9\.]+)"]

  for each in patterns:
    ro = re.match(each, line)
    if ro is not None:
      return (float(ro.group(1)), float(ro.group(2)), float(ro.group(3)))

  return None



TRAIN_START = "train_start"
TRAIN_END = "train_end"
SCRIPT_START = "script_start"
EPOCH_START = "epoch_start"
EPOCH_END = "epoch_end"
  
def parse_log(fname, output_timestamp):
  timeline = []
  processed_timeline = {}
  elastic_metrics = []
  loss_metrics = []

  def init_dict(key, init_type=0):
    if(key not in processed_timeline):
      processed_timeline[key] = [] if init_type == 0 else {}

  epoch_starts = {}
  with open(fname, 'r') as fp:
    for line in fp.readlines():
      if("Training begins" in line):
        init_dict(TRAIN_START)
        processed_timeline[TRAIN_START].append(match_timestamp_lines(line))
        timeline.append((TRAIN_START, match_timestamp_lines(line)))
      elif("Finished Training" in line):
        init_dict(TRAIN_END)
        processed_timeline[TRAIN_END].append(match_timestamp_lines(line))
        timeline.append((TRAIN_END, match_timestamp_lines(line)))
      elif("Script starts" in line):
        init_dict(SCRIPT_START)
        processed_timeline[SCRIPT_START].append(match_timestamp_lines(line))
        timeline.append((SCRIPT_START, match_timestamp_lines(line)))
      elif("Epoch" in line and "begins" in line):
        init_dict(EPOCH_START, 1)
        tmp = match_epoch_lines(line)
        processed_timeline[EPOCH_START][tmp[0]] = tmp[1]
        event = f"epoch{tmp[0]}_start"
        timeline.append((event, tmp[1]))
        if(event not in epoch_starts):
          epoch_starts[event] = []
        epoch_starts[event].append(tmp[1])
        
      elif("Epoch" in line and "ends" in line):
        init_dict(EPOCH_END, 1)
        tmp = match_epoch_lines(line)
        processed_timeline[EPOCH_END][tmp[0]] = tmp[1]
        timeline.append((f"epoch{tmp[0]}_end", tmp[1]))

      elif("tensorflowelastic" in line and "_initialize_workers.duration.ms" in line):
        elastic_metrics.append((f"initialize_workers took {match_elatic_metrics(line)}"))

      elif(all([x in line for x in ["loss:", "accuracy:", "top_5_accuracy:"]])):
  #      print(line)
        tmp_loss_metrics = match_loss_metrics(line)
        assert tmp_loss_metrics is not None
        loss_metrics.append(tmp_loss_metrics)


  
  stimeline = sorted(timeline, key=lambda event: event[1])
  
  #Now we infer from the keys when to get duration
  timeline_str = ""
  total_train_start = 0
  last_train_end = 0
  
  def is_epoch_start(event):
    return "epoch" in event[0] and "start" in event[0]
  def is_epoch_end(event):
    return "epoch" in event[0] and "end" in event[0]
  def output_event(event):
    return f"{event[0]}:{event[1]}\n" if output_timestamp else ""
  
  for ii, event in enumerate(stimeline):

    if(event[0] == SCRIPT_START):
      #before the script starts we will pop off the first initialize_workers
      if(len(elastic_metrics) > 0):
        timeline_str += f"{elastic_metrics.pop(0)}\n"
    if ii == 0:
      timeline_str +=output_event(event)
      total_train_start = event[1]
      continue
    
    if(event[0] == TRAIN_START and stimeline[ii-1][0] == SCRIPT_START):
      timeline_str +=f"Script setup took {event[1]-stimeline[ii-1][1]}\n"
      timeline_str +=output_event(event)

    if(is_epoch_start(event)):
      if(ii > 2 and stimeline[ii-1][0]==TRAIN_START and stimeline[ii-2][0]==SCRIPT_START and is_epoch_start(stimeline[ii-3])):
        timeline_str += f"Epoch time lost {event[1]-stimeline[ii-3][1]}\n"
      timeline_str +=output_event(event)
      
      
    if(is_epoch_end(event)):
      first_of_epoch = sorted(epoch_starts[event[0].replace("end","start")])[0]
      epoch = int(event[0].replace("_end","").replace("epoch",""))
      print(epoch)
      print(len(loss_metrics), epoch)
      timeline_str+=f"Epoch {epoch} took {event[1] - first_of_epoch:.2f} loss: {loss_metrics[epoch][0]} accuracy: {loss_metrics[epoch][1]} top_5_accuracy: {loss_metrics[epoch][2]}\n"
      #timeline_str+=f"Epoch {epoch}: loss: {loss_metrics[epoch][0]} accuracy: {loss_metrics[epoch][1]} top_5_accuracy: {loss_metrics[epoch][2]}\n"
      timeline_str +=output_event(event)
    
    if(event[0] == TRAIN_END):
      timeline_str +=output_event(event)
      last_train_end = event[1]

  timeline_str += f"Total train time is {last_train_end - total_train_start}\n"
  print(timeline_str)

    



def parse_logs(filelist, output_timestamp):
  for fname in filelist:
    parse_log(fname, output_timestamp)

def main(args=None):
  args = parse_args(args)

  filelist = []
  if(args.log_dir):
    if(os.path.isdir(args.log_dir)):
      filelist += [x for x in os.listdir(args.log_dir)]
    else:
      raise ValueError(f"{args.log_dir} is not a directory")
  elif(args.log_file):
    if(os.path.isfile(args.log_file)):
      filelist = [args.log_file]
    else:
      raise ValueError(f"{args.log_file} is not a file")
  else:
    raise ValueError(f"either --log_file or --log_dir need to be set")
  output_timestamp = args.output_timestamp if args.output_timestamp is not None else True
  parse_logs(filelist, output_timestamp)


if __name__ == "__main__":
  main()
