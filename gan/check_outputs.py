#!/usr/bin/env python3
import json
import os
import sys
[print(f, " ", len(json.loads(open(os.path.join(sys.argv[1], f), "r", encoding="utf8").read()))) for f in os.listdir(sys.argv[1])]
