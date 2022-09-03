#!/usr/bin/python3
# -*- coding: utf-8 -*-
from pylogging import loggings

mylogging = loggings('utils').getlog()

def func1(params):
    print('get params%s', params)
    mylogging.info('get params%s', params)
    return 0