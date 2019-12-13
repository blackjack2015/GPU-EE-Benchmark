import time, argparse
import datetime
import sys,urllib,urllib2
import csv
import os, glob, re
import cPickle as pickle
import numpy as np
import ConfigParser
import json
import pandas as pd

#fp = "resnet50-n1-bs16-lr0.0100-ns1/"
#fp = "lstm-n1-bs256-lr1.0000-ns1/"
#fp = "lstman4-n1-bs32-lr0.0003-ns1/"
fp = "ssd-n1-bs32-lr0.0010-ns1/"

f = open(fp+"hpclgpu-power.log", 'r')
content = f.readlines()[2:-3]
f.close()

coreF=1380
powerList = [float(line.split()[-1].strip()) / 1000.0 for line in content if int(line.split()[3]) == coreF]
#powerList = [float(line.split()[-1].strip()) / 1000.0 for line in content]

powerList = powerList[len(powerList) / 10 * 5 :len(powerList) / 10 * 6]   # filter out those power data of cooling down GPU
powerList.sort()
sampleLen = len(powerList) / 2
powerList = powerList[sampleLen:]   # filter out those power data of cooling down GPU
print np.mean(powerList)

f = open(fp+"hpclgpu.log", 'r')
content = f.readlines()[2:-3]
f.close()

perfList = [float(line.split()[-2].strip()) for line in content if 'Speed' in line]

perfList = perfList[-20:]
print np.mean(perfList)



