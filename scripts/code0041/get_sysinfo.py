#!/usr/bin/env python
#coding=utf-8
import os
mem=[]
net=[]

def dump(content):
    for id, item in enumerate(content, 1):
        print("%d:" % id)
        for k in item:
            print(" %s: %s" % (k, item[k]))
    print("\n")

def parse(content):
    info = {}
    for line in content:
        t = line.split(": ")
        if len(t) < 2:
            continue
        name = t[0].strip()
        value = t[1].strip()
        info[name] = value
    return info

data = os.popen("/usr/sbin/dmidecode").read()
sectors = data.split("\n\n")
for sector in sectors:
    lines = sector.split("\n")
    if len(lines) < 3:
        continue
    title = lines[1]
    info = parse(lines)
    if title == 'Memory Device':
        if not info['Size'].startswith('No'):
            mem.append({'size':info['Size']})

data = os.popen("/sbin/ip -o -f inet addr").read()
lines = data.split("\n")
for line in lines:
    items = line.split()
    if len(items) < 4:
        continue
    net.append({'name':items[1],'addr':items[3]})


print("[OS]")
os.system('/bin/uname -nsr')
print(open('/etc/issue.net').readline())

print("[CPU]")
lines=open('/proc/cpuinfo').readlines()
info=parse(lines)
print("Model Name: %s\nAddress sizes: %s\n" % (info['model name'], info['address sizes']))
os.system('/usr/bin/lscpu')
print("\n")

print("[Memory]")
lines=open('/proc/meminfo').readlines()
info=parse(lines)
print("total: %s ; swap: %s" % (info['MemTotal'], info['SwapTotal']))
dump(mem)

print("[FS]")
os.system('/bin/lsblk')
print("")
os.system('/bin/df -h')

print("\n")
print("[Net]")
dump(net)