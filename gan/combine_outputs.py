#!/usr/bin/env python3
import re
import sys
import os
import json

if len(sys.argv) == 2:
  data = []
  for output_file in os.listdir(sys.argv[1]):
    output_fd = open(os.path.join(sys.argv[1], output_file), "r", encoding="utf8")
    data += json.loads(output_fd.read())
    output_fd.close()
  data_dict = dict([(d["key"], d) for d in data])
  data_filtered = list(data_dict.values())
  print(data_filtered[0])
  print("filtered: %f -> %d" % (len(data_filtered)/len(data), len(data_filtered)))
  output_combined_fd = open("output_combined", "w")
  output_combined_fd.write(json.dumps(data_filtered))
  output_combined_fd.close()
  data_dir = "data_filtered_thres_10.0"
  if os.path.isdir(data_dir):
    key_regex_output = re.compile("([^\/]+)\/$")
    key_regex_data = re.compile("(.+).jpg$")
    keys_data = set([key_regex_data.findall(key)[0] for key in os.listdir(data_dir)])
    output_combined_filtered = list(filter(lambda key: key_regex_output.findall(key["key"])[0] in keys_data, data_filtered))
    print("filtered by data: %f -> %d" % (len(output_combined_filtered)/len(data_filtered), len(output_combined_filtered)))
    output_combined_fd = open("output_combined_filtered", "w")
    output_combined_fd.write(json.dumps(output_combined_filtered))
    output_combined_fd.close()

    


  
    
