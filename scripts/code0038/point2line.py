import math

def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    #这里的XY代表要求的点，（x1，y1）（x2，y2）是用来确定直线用的
    a = lineY2-lineY1
    b = lineX2-lineX1
    c = lineX2*lineY1-lineX1*lineY2
    dis = float((a*pointX+b*pointY+c)/(math.pow(a*a+b*b,0.5)))
    #注意：这里没有加绝对值，得出的数有正负之分
    #pow--根号下
    return dis
#定义点到线距离函数,返回一个距离

line1_x1 = float(input('input lin1_x1:'))
line1_y1 = float(input('input lin1_y1:'))
line1_x2 = float(input('input lin1_x2:'))
line1_y2 = float(input('input lin1_y2:'))

data = []

for i in range (3):
    data_input_x = float(input('input x:'))
    data_input_y = float(input('input y:'))
    data_input = getDis(data_input_x,data_input_y,line1_x1,line1_y1,line1_x2,line1_y2)
    data.append(data_input)
    print(data)
