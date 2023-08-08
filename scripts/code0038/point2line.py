import math

def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    # 已知直线上的两点P1(X1,Y1) P2(X2,Y2)， P1 P2两点不重合。则直线的一般式方程AX+BY+C=0中，A B C分别等于：
    a = lineY2-lineY1
    b = lineX2-lineX1
    # c = lineX2*lineY1-lineX1*lineY2
    c = lineX1*(lineY1-lineY2) + lineY2*(lineX2-lineX1)
    dis = float((a*pointX+b*pointY+c)/(math.pow(a*a+b*b,0.5)))
    return dis

