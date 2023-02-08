# clss_list = [u'r_light_off', u'lcd_meter', u'r_light_off', u'g_light_off', \
#         u'pressure_meter', u'lcd_meter', u'g_light_off', u'lcd_meter', \
#         u'lcd_meter', u'g_light_off', u'r_light_off', u'g_light_off', \
#         u'r_light_off', u'r_light_off', u'g_light_off']

# resultlist = [u'r_light_off', u'lcd_meter', u'r_light_off', u'g_light_off', \
#         u'pressure_meter', u'lcd_meter', u'g_light_off', u'lcd_meter', \
#         u'lcd_meter', u'g_light_off', u'r_light_off', u'g_light_off', \
#         u'r_light_off', u'r_light_off', u'g_light_off']

# detect_type = ['r_light_on', 'r_light_off', 'g_light_on', 'g_light_off']

# final_resultlist = []

# for i, item in enumerate(clss_list):  # 判断obj是否在列表里，在的话则将结果加到finalresultlist
#     if str(item) in str(detect_type):
#         for j, element in enumerate(resultlist):
#                 if i == j:
#                     print(i, j, element, type(element))
#                     final_resultlist.append(element)
#     elif str(item) not in str(detect_type):
#         pass
#     else:
#         pass
# print(final_resultlist, len(final_resultlist))
i = 5
for s in range(1,i):
    print(s)