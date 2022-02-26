# -*- coding: utf-8 -*-

'''
例如：list_a=[1,5,2,3]，list_b=[2,3,1,4]保持下标对应，即 lsit_a中 1 对应 list_b中 2，
以此类推，list_a升序后 list_sort=[1, 2, 3, 5]，lsit_sort中 1 对应 2 这一关系仍保持不变
'''

list_a = [1,5,2,3]
list_b = [2,3,1,4]
list_reflect = []
list_sort = sorted(list_a, key=int, reverse=False)
 
print("list_a = ", list_a)
print("list_b = ", list_b)
print("list_sort = ", list_sort)
 
for x in list_sort:
    if x in list_a:
        list_reflect.append(list_b[list_a.index(x)])
print("list_reflect = ", list_reflect)