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

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-setting', type=str, help='gpu benchmark setting', default='v100')
parser.add_argument('--kernel-setting', type=str, help='kernels of benchmark', default='A')
parser.add_argument('--core-base', type=int, help='base core frequency', default=0)
parser.add_argument('--mem-base', type=int, help='base memory frequency', default=0)

opt = parser.parse_args()
print opt

gpucard = opt.benchmark_setting
version = opt.kernel_setting
coreBase = opt.core_base
memBase = opt.mem_base

logRoot = 'logs/%s-%s' %( gpucard, version)

power_filelist = glob.glob(r'%s/*power.log' % logRoot)
power_filelist.sort()

# Reading metrics list
cf_bs = ConfigParser.SafeConfigParser()
cf_bs.read("configs/benchmarks/%s.cfg" % (opt.benchmark_setting))
coreBase = json.loads(cf_bs.get("dvfs_control", "coreBase"))
memBase = json.loads(cf_bs.get("dvfs_control", "memBase"))
powerState = json.loads(cf_bs.get("dvfs_control", "powerState"))

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("configs/kernels/%s.cfg" % opt.kernel_setting)
benchmark_programs = cf_ks.sections()

head = ["appName", "coreF", "memF", "argNo", "kernel", "power/W"]
print head

# prepare csv file
csvfile = open('./%s-%s-Power.csv' % (opt.benchmark_setting, opt.kernel_setting), 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

dvfsEnv = 'linux'
for fp in power_filelist:
    # print fp

    baseInfo = fp.split('_')
    appName = baseInfo[1]
    coreF = coreBase
    memF = memBase
    argNo = baseInfo[2]

    kernel = json.loads(cf_ks.get(appName, 'kernels'))[0]
    rec = [appName, coreF, memF, argNo, kernel]

    # neglect first two lines of device information and table header
    f = open(fp, 'r')
    content = f.readlines()[2:]
    f.close()
    
    if dvfsEnv == 'linux': # filter with frequency
        print coreF
        powerList = [float(line.split()[-1].strip()) / 1000.0 for line in content if int(line.split()[3]) == coreF]
    else:
        powerList = [float(line.split()[-1].strip()) / 1000.0 for line in content if int(line.split()[1]) == runState]

    #powerList = powerList[len(powerList) / 10 * 5 :len(powerList) / 10 * 6]   # filter out those power data of cooling down GPU
    powerList.sort()
    sampleLen = len(powerList) / 4
    powerList = powerList[-sampleLen:]   # filter out those power data of cooling down GPU
    rec.append(np.mean(powerList))

    print rec
    csvWriter.writerow(rec[:len(head)])

