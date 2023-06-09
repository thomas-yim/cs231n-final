import sys
import numpy as np
import matplotlib.pyplot as plt
import re

USER = "Caroline"

if len(sys.argv) > 1:
    IN_FILE = sys.argv[1]
elif USER == "Caroline":
    IN_FILE = "/Users/carolinecahilly/Desktop/loss_log.txt"

if USER == "Caroline":
    OUT_PATH = "" # current directory

with open(IN_FILE, 'r') as file:
    data = file.read()

matches = re.findall(r'^={16,}\n', data, re.MULTILINE)

# Get the data after each matching line
data_list = []
for match in matches:
    data = data.split(match)[-1].strip()
    data_list.append(data)

print(data_list)