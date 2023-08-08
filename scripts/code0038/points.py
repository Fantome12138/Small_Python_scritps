def isinpolygon(point, vertex_lst):
    '''
    检测点是否位于区域外接矩形内
    :param point: 目标点
    :param vertex_lst: 围成的区域集合
    :param contain_boundary:
    :return:判断点是否在外接区域内，如果不在，直接返回false
    '''
    lngaxis, lataxis = zip(*vertex_lst) # lngaxis代表x轴的元组集合，lataxis代表y轴的元组集合
    minlng, maxlng = min(lngaxis), max(lngaxis) # minlng--x轴元组最小值，maxlng--x轴元组最大值
    minlat, maxlat = min(lataxis), max(lataxis) # minlat--y轴元组最小值，maxlat--y轴元组最大值
    lng, lat = point # lng--目标点x值, lat--目标点y值
    isin = (minlng <= lng <= maxlng) & (minlat <= lat <= maxlat)

    return isin

def isintersect(poi, spoi, epoi):
    '''
    输入：判断点，边起点，边终点，都是[lng,lat]格式数组, 射线是与x轴水平线
    poi : 目标点
    spoi: 第1个边界坐标到最后一个坐标
    epoi：第0个边界坐标到倒数第一个坐标
    '''
    lng, lat = poi    # lng--目标点X轴, lat--目标点Y轴
    slng, slat = spoi # slng--边界点的X轴, slat--边界点的Y轴
    elng, elat = epoi # elng--边界点的X轴, elat--边界点的Y轴
    if poi == spoi:
        # print("在顶点上")
        return None
    if slat == elat:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if slat > lat and elat > lat: # 线段在射线上边
        return False
    if slat < lat and elat < lat: # 线段在射线下边
        return False
    if slat == lat and elat > lat: # 交点为上端点，对应spoint
        return False
    if elat == lat and slat > lat: # 交点为下端点，对应epoint
        return False
    if slng < lng and elat < lat: # 线段在射线左边
        return False
    # 求交点
    xseg = elng-(elng-slng)*(elat-lat)/(elat-slat)
    if xseg == lng:
        #print("点在多边形的边上")
        return None
    if xseg < lng: # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后

def isin_multipolygon(poi, vertex_lst):
    '''
    :param poi: 需要判断的目标点
    :param vertex_lst: 区域点的集合
    :return:范围TRUE和FLASE，如果穿过点个数为偶数，说明点在区域外
    '''
    if not isinpolygon(poi, vertex_lst): # 判断点是否在外接区域内
        return False
    sinsc = 0
    for spoi, epoi in zip(vertex_lst[1:], vertex_lst[:-1]):
        print(spoi, epoi)
        '''
        spoi: 第1个边界坐标到最后一个坐标
        epoi：第0个边界坐标到倒数第一个坐标
        先判断是不是在线段上，如果是，直接返回TRUE
        如果不是在线段上，判断射线和整个多边形线段的True的个数，即角点个数，如果交点个数为奇数，说明在多边形内，如果为偶数，说明不在
        '''
        intersect = isintersect(poi, spoi, epoi)
        if intersect is None: # 点在线段上，直接返回True，不做其他循环了
            return True
        elif intersect is True: # 如果返回来的是true，计数+1
            sinsc += 1
    return sinsc % 2 == 1 # 如果是2的倍数，返回False，不是2的倍数，返回false


def isPointinPolygon(point, rangelist):
    # 判断是否在外包矩形内，如果不在，直接返回false
    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    if (point[0] > maxlng or point[0] < minlng or
        point[1] > maxlat or point[1] < minlat):
        return False
    
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        print(f'point2: {point2} ')
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            print(point12lng, point[0])
            # 点在多边形边上
            if (point12lng == point[0]):
                print("点在多边形边上")
                return False
            if (point12lng < point[0]):
                count +=1
        point1 = point2
    print(count)
    if count%2 == 0:
        return False
    else:
        return True


if __name__ == '__main__':
    # vertex_lst = [[1, 0], [2, 1], [1, 2], [0, 1], [1, 0]]
    # poi = [0.5, 0.5]
    # print(isin_multipolygon(poi, vertex_lst))
    
    # 下面这个函数更好理解
    print(isPointinPolygon([0.4,0.6], [[0,0],[1,1],[0,1],[0,0]]))
    

