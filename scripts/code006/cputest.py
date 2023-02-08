import time

starttime = time.process_time()

def doubleFact(x):
    ans = 1
    for i in range(1, x+1):
        if i%2 == x%2:
            ans *= i
    return ans
    
def asin(x, t):
    answer = 0
    for k in range(0, t+1):
        a = (doubleFact(2*k-1) / doubleFact(2*k)) * (pow(x, 2*k+1)/(2*k+1))
        print("k=%d, a=%s"%(k, a))
        answer += a
 
    return answer
    
print(asin(1, 3000)*2)
endtime = time.process_time()
print('-------------')
print (endtime-starttime)


# t = 3000时，腾讯云双核cpu速度6s，surfacebook2速度11s



