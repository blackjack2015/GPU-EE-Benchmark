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
parser.add_argument('--benchmark-setting', type=str, help='gpu benchmark setting', default='p100')
parser.add_argument('--kernel-setting', type=str, help='kernels of benchmark', default='synthetic')
parser.add_argument('--core-base', type=int, help='base core frequency', default=0)
parser.add_argument('--mem-base', type=int, help='base memory frequency', default=0)

opt = parser.parse_args()
print opt

gpucard = opt.benchmark_setting
version = opt.kernel_setting
coreBase = opt.core_base
memBase = opt.mem_base

logRoot = 'logs/%s-%s' %( gpucard, version)

perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)
metrics_filelist = glob.glob(r'%s/*metrics.log' % logRoot)

perf_filelist.sort()
metrics_filelist.sort()

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("configs/kernels/%s.cfg" % version)
benchmark_programs = cf_ks.sections()

metrics = []
recs = []
for fp in perf_filelist:
    fn = fp.split('/')[-1]
    print fn

    baseInfo = fn.split('_')
    appName = baseInfo[1]
    print baseInfo
    coreF = str(int(baseInfo[2][4:]) + coreBase)
    memF = str(int(baseInfo[3][3:]) + memBase)
    argNo = baseInfo[4]

    kernel = json.loads(cf_ks.get(appName, 'kernels'))[0]
    rec = [appName, coreF, memF, argNo, kernel]

    # extract execution time information
    f = open(fp, 'r')
    content = f.readlines()
    f.close()
    print fp
    regex = re.compile(r'.*\%.*' + kernel)
    time = filter(regex.search, content)[0].split()[3].strip()
    if 'us' in time:
        time = float(time[:-2]) / 1000
    else:
        time = float(time[:-2])

    isLog = False
    if isLog:
        regex = re.compile(r'(iterated \d+, average time is)|(Average Kernel Time)|(Average Time)')
        timeRaw = filter(regex.search, content)
        if len(timeRaw) == 0:
            continue
        time = float(timeRaw[0].split()[-2].strip())
        print time
    else:  
        regex = re.compile(r'.*\%.*' + kernel)
        time = filter(regex.search, content)[0].split()[3].strip()
        if 'us' in time:
            time = float(time[:-2]) / 1000
        elif 'ms' in time:
            time = float(time[:-2])
        else:
	    time = float(time[:-1]) * 1000
        print time
    rec.append(time)

    # extract grid and block settings
    fm, number = re.subn('perf', 'metrics', fp)
    f = open(fm, 'r')
    content = f.readlines()
    maxLen = len(content)
    f.close()
    regex = re.compile(r'\(\d+ \d+ \d+\).*(void )*' + kernel + r'[<(]+')
    message = [value \
                for s, value in enumerate(content) \
                    for m in [regex.search(value)] if m]
    if len(message) != 0:
        message = message[0]
        # print message
        grid_block = re.findall(r'\(\d+ \d+ \d+\)', message)
        grid_block = " ".join(grid_block)
        warps_info = [int(item) for item in grid_block.translate(None, '()').split()]
        warps = float(np.prod(np.array(warps_info))) / 32.0
        print grid_block, warps
        rec.append(grid_block)
        rec.append(warps)
    else:
        rec.append(" ")
        rec.append(" ")

    # extract metrics information
    stride = 2
    fm, number = re.subn('perf', 'metrics', fp)
    f = open(fm, 'r')
    content = f.readlines()
    maxLen = len(content)
    f.close()
    regex = re.compile(r'Kernel: (void )*' + kernel + r'[<(]+')
    message = [line.strip() \
                for s, value in enumerate(content) \
                    for m in [regex.search(value)] if m \
                for line in content[s+1:min(maxLen,s+1+stride)]]

    print len(message)
    if len(message) == 46:
        message = message[:-1]

    for line in message:
        metric = line.split()[1]

        # initialize metrics list
        if len(metrics) < len(message):
            metrics.append(metric)

        value = line.split()[-1]

        if '%' in value:
            value = float(value[:-1]) / 100
        elif 'TB/s' in value:
            value = float(value[:-4]) * 1e3
        elif 'GB/s' in value:
            value = float(value[:-4])
        elif 'MB/s' in value:
            value = float(value[:-4]) * 1e-3
        elif 'KB/s' in value:
            value = float(value[:-4]) * 1e-6
        elif 'B/s' in value:
                value = float(value[:-3]) * 1e-9

        rec.append(value)

    # print rec
    recs.append(rec)
    # csvWriter.writerow(rec[:len(head)])

head = ["appName", "coreF", "memF", "argNo", "kernel", "time/ms", "blocks", "warps"] + metrics
print head

# prepare csv file
csvfile = open('csvs/%s-%s-Performance.csv' % (gpucard, version), 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

# write records
for rec in recs:
    csvWriter.writerow(rec[:len(head)])

# tempf = open('perfData.bin', 'wb')
# pickle.dump(record, tempf, 0)
# tempf.close()

if not os.path.exists("csvs/raw"):
    os.makedirs("csvs/raw")

logRoot = 'logs'
csvRoot = 'csvs/raw'

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-setting', type=str, help='gpu benchmark setting', default='gtx980-high-dvfs')
parser.add_argument('--kernel-setting', type=str, help='kernels of benchmark', default='real-small-workload')

opt = parser.parse_args()
print opt

logRoot = "%s/%s-%s" % (logRoot, opt.benchmark_setting, opt.kernel_setting)

power_filelist = glob.glob(r'%s/*power.log' % logRoot)
power_filelist.sort()

# Reading metrics list
cf_bs = ConfigParser.SafeConfigParser()
cf_bs.read("configs/benchmarks/%s.cfg" % (opt.benchmark_setting))
metrics = json.loads(cf_bs.get("profile_control", "metrics"))
coreBase = json.loads(cf_bs.get("dvfs_control", "coreBase"))
memBase = json.loads(cf_bs.get("dvfs_control", "memBase"))
powerState = json.loads(cf_bs.get("dvfs_control", "powerState"))
runState = json.loads(cf_bs.get("dvfs_control", "runState"))
dvfsEnv = json.loads(cf_bs.get("dvfs_control", "dvfsEnv"))

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("configs/kernels/%s.cfg" % opt.kernel_setting)
benchmark_programs = cf_ks.sections()

head = ["appName", "coreF", "memF", "argNo", "kernel", "power/W"]
print head

# prepare csv file
csvfile = open('%s/%s-%s-Power.csv' % (csvRoot, opt.benchmark_setting, opt.kernel_setting), 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

for fp in power_filelist:
    # print fp

    baseInfo = fp.split('_')
    appName = baseInfo[1]
    coreF = int(baseInfo[2][4:]) + coreBase
    memF = int(baseInfo[3][3:]) + memBase
    argNo = baseInfo[4]

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
    powerList = powerList[-100:]   # filter out those power data of cooling down GPU
    rec.append(np.mean(powerList))

    print rec
    csvWriter.writerow(rec[:len(head)])

