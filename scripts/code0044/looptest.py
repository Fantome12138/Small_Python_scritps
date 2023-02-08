#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: wxnacy(wxnacy@gmail.com)
# Description: for while generator list_comprehension map 对比速度


'''
速度：  map > 列表解析 > 生成器 > for > while

`map` 是内置函数，底层由 C 来编写，最快是毫无疑问的。而 `while` 是纯 Python 实现的，所以性能最差。

这次我们可以下个结论，处理循环时，我们已经尽可能的使用内置方法，然后根据业务需求来选择使用列表解析和生成器，实在不行了使用 `for` 循环，而 `while` 则是尽量不去使用的。

'''
def loop_for(n):
    res = []
    for i in range(n):
        res.append(abs(i))
    return res

def loop_while(n):
    i = 0
    res = []
    while i < n:
        res.append(abs(i))
        i += 1
    return res

def loop_generator(n):
    '''使用生成器'''
    res = (abs(i) for i in range(n))
    res =  list(res)
    return res

def loop_list_compre(n):
    '''使用列表解析'''
    res = [abs(i) for i in range(n)]
    return res

def loop_map(n):
    '''使用内置函数 map'''
    return list(map(abs, range(n)))

import unittest

class TestMain(unittest.TestCase):

    def setUp(self):
        '''before each test function'''
        pass

    def tearDown(self):
        '''after each test function'''
        pass

    def test_func(self):
        n = 10
        # 我们要求他们生成的结果是一样的
        flag = (loop_for(n) == loop_while(n) == loop_generator(n) ==
                loop_list_compre(n) == loop_map(n))
        self.assertTrue(flag)

import timeit

def print_func_run_time(count, func, **kw):
    b = timeit.default_timer()
    for i in range(count):
        func(**kw)
    print('{} run {} times used {}s'.format(
        func.__name__.ljust(20),
        count,
        timeit.default_timer() -b ))

if __name__ == "__main__":
    count = 1000
    n = 1000
    print_func_run_time(count, loop_for, n = n)
    print_func_run_time(count, loop_while, n = n)
    print_func_run_time(count, loop_generator, n = n)
    print_func_run_time(count, loop_list_compre, n = n)
    print_func_run_time(count, loop_map, n = n)
    unittest.main()

# ----------------------------------------------------------------------
# Ran 1 test in 0.000s
#
# OK
# loop_for             run 1000 times used 0.14018906400087872s
# loop_while           run 1000 times used 0.21399457900042762s
# loop_generator       run 1000 times used 0.12857274799898732s
# loop_list_compre     run 1000 times used 0.08585307099929196s
# loop_map             run 1000 times used 0.043123570998432115s