import numpy as np

def relu(t):
    res = np.copy(t)
    res[t < 0] = 0
    return res

def drelu(t):
    res = np.copy(t)
    res[t > 0] = 1
    res[t <= 0] = 0
    return res

def softmax(X):
    for i in range(0, len(X)):
        X[i,:] = X[i,:] - np.max(X[i,:])
        X[i,:] = np.exp(X[i, :]) / (np.sum(np.exp(X[i, :])))
    return X

def gradient_clip(dw,min,max):
    res = np.copy(dw)
    res[dw<min] = min
    res[dw>max] = max
    return res
# 该卷积网络层次结构

def img2col_conv(X, filter, step):
    '''
    :param X: 输入 [1,28,28,3]
    :param filter: 卷积核 [1,3,3,3]
    :param step:  1
    :param padding: 0
    :return:
    '''
    f_b, f_h, f_w, f_c = filter.shape
    filter_convert = np.zeros(shape=[f_w * f_h * f_c, f_b])
    for b in range(0,f_b):
        for c in range(0,f_c):
            f_unit = filter[b,:,:,c].flatten()
            star_p = c * len(f_unit)
            end_p = star_p + len(f_unit)
            filter_convert[star_p:end_p,b] = f_unit
    cur = 0
    height_out, width_out = int(np.ceil((X.shape[1] - filter.shape[1] + 1) / step)), int(
        np.ceil((X.shape[2] - filter.shape[2] + 1) / step))
    x_convert = np.zeros(shape=[width_out * height_out * X.shape[0], f_h * f_w * f_c])
    for b in range(0,X.shape[0]):
        for y in range(0,X.shape[1]-filter.shape[1]+1,step):
            for x in range(0,X.shape[2]-filter.shape[2]+1,step):
                for c in range(0,X.shape[3]):
                    tile = X[b,y:y + f_h, x:x + f_w, c]
                    star_p = c * f_h * f_w
                    end_p = star_p + f_h * f_w
                    x_convert[cur,star_p:end_p] = tile.flatten()
                cur = cur + 1
    state = np.dot(x_convert,filter_convert)
    res = np.zeros(shape=[X.shape[0],height_out,width_out,f_b])
    for b in range(0,res.shape[0]):
        star_p = b * width_out * height_out
        end_p =star_p + width_out * height_out
        for c in range(0,f_b):
            tile = state[star_p:end_p,c].reshape(height_out,width_out)
            res[b,:,:,c] = tile
    return x_convert,filter_convert,state,res

def img2col_maxpool(X,pool_size,step):
    height_out,width_out = int(np.ceil((X.shape[1] - pool_size[0] + 1) / step)), int(
        np.ceil((X.shape[2] - pool_size[1] + 1) / step))
    pool_convert = np.zeros(shape=[height_out * width_out * X.shape[0],pool_size[0] * pool_size[1],X.shape[3]])
    pool_height,pool_width = pool_size
    cur = 0
    for b in range(0,X.shape[0]):
        for y in range(0,X.shape[1]-pool_height+1,step):
            for x in range(0,X.shape[2]-pool_width+1,step):
                tile = X[b,y:y + pool_height , x:x + pool_width]
                for c in range(0,X.shape[3]):
                    pool_convert[cur,:,c] = tile[:,:,c].flatten()
                cur = cur + 1
    index = np.argmax(pool_convert,axis=1)
    p_c = np.zeros_like(index,dtype=float)
    for y in range(0,p_c.shape[0]):
        for c in range(0,p_c.shape[1]):
            p_c[y,c] = pool_convert[y,index[y,c],c]
    res = np.zeros(shape=[X.shape[0],height_out,width_out,X.shape[3]])
    for b in range(0,res.shape[0]):
        start_p =b * (width_out * height_out)
        end_p = start_p + (width_out * height_out)
        for c in range(0,res.shape[3]):
            tile = p_c[start_p:end_p,c].reshape(height_out,width_out)
            res[b,:,:,c] = tile
    return pool_convert,p_c,index,res

def conv_flatten(x_flatten,os):
    res = np.zeros(shape = os)
    for i in range(0,len(x_flatten)):
        for c in range(0,os[3]):
            start_p = c * os[1] * os[2]
            end_p = start_p + os[1] * os[1]
            res[i,:,:,c] = x_flatten[i,start_p:end_p].reshape(os[1],os[2])
    return res
def flatten(x_pool2):
    x_flatten = np.zeros(shape=[x_pool2.shape[0],x_pool2.shape[1] * x_pool2.shape[2] * x_pool2.shape[3]])
    for i in range(0,x_flatten.shape[0]):
        for c in range(0,x_pool2.shape[3]):
            start_p = c * (x_pool2.shape[1] * x_pool2.shape[2])
            end_p =start_p + (x_pool2.shape[1] * x_pool2.shape[2])
            x_flatten[i,start_p:end_p] = x_pool2[i,:,:,c].flatten()
    return x_flatten
def entrop_loss(y_p,y_label):
    return np.mean(np.sum(-y_label * np.log(y_p+1e-5),axis=1))
def forward(X,Paramters):
    filter1,filter2,w3,w4 = Paramters
    # 第一层：卷积层
    x_convet1,filter_convert1,state1,x_conv1=img2col_conv(X,filter1,1)
    a_1 = relu(x_conv1)
    cash1 = {'z_p':X,'a_p':X,'z':x_conv1,'a':a_1,'w':filter1.copy()}
    # 第二次：池化层
    cv_p1,p_c1,index1,x_pool1 = img2col_maxpool(cash1['a'],(2,2),2)
    cash2 = {'z_p':cash1['z'],'a_p':cash1['a'],'z':x_pool1,'a':x_pool1,'w':(2,2),'os':x_pool1.shape,'index':index1}

    # 第三层：卷积层
    x_convet2, filter_convert2, state2, x_conv2 = img2col_conv(x_pool1,filter2,step=1)
    a_2 = relu(x_conv2)
    cash3 = {'c_z_p':state2,'c_a_p':x_convet2,'c_w':filter_convert2,'z_p':cash2['z'],'a_p':cash2['a'],'z':x_conv2,'a':a_2,'w':filter2.copy()}

    # 第四层：池化层
    cv_p2,p_c2,index2,x_pool2 = img2col_maxpool(x_conv2,(2,2),2)
    cash4 = {'z_p':cash3['z'],'a_p':cash3['a'],'z':x_pool2,'a':x_pool2,'w':(2,2),'os':x_pool2.shape,'index':index2}
    # 第五层: 隐藏层
    x_flatten = flatten(x_pool2)
    f3 = np.dot(x_flatten,w3)
    a_3 = relu(f3)
    cash5 = {'z_p':x_flatten,'a_p':x_flatten,'z':f3,'a':a_3,'w':w3.copy()}
    # 输出层
    f4 = np.dot(f3,w4)
    y_p = softmax(f4)
    cash6 = {'z_p':cash5['z'],'a_p':cash5['a'],'z':f4,'a':y_p,'w':w4.copy()}
    return [cash1,cash2,cash3,cash4,cash5,cash6],y_p

# 全连接层的反向传播
def full_backprop(delta,cash):
    dw = np.dot(cash['a_p'].T,delta)
    db = np.sum(delta,axis=0)
    delta_pre = np.dot(delta,cash['w'].T) * drelu(cash['z_p'])
    grad_dict = {'dw':dw,'db':db,'delta_pre':delta_pre}
    return grad_dict

#计算池化层的反向传播:
def upsample(delta,poos_size,target_shape,index):
    res = np.zeros(shape=target_shape,dtype=float)
    cur = 0
    for b in range(0,target_shape[0]):
        for y in range(0,target_shape[1] - poos_size[0] + 1,poos_size[0]):
            for x in range(0,target_shape[2] - poos_size[0] + 1,poos_size[1]):
                for c in range(target_shape[3]):
                    i = index[cur,c]
                    x_epoch = i % poos_size[1]
                    y_epoch = int(i / poos_size[0])
                    res[b,y+y_epoch,x+x_epoch,c] = delta[b,int(y/poos_size[0]),int(x/poos_size[0]),c]
                cur = cur + 1
    return res
def pool_backprop(delta_pool,cash,flattened = True):
    if flattened:
        delta_pool = conv_flatten(delta_pool,cash['os'])
    return upsample(delta_pool,cash['w'],cash['z_p'].shape,cash['index'])
def swap_first_end_axis(mat):
    delta = np.copy(mat)
    delta = np.rollaxis(delta,3,0)
    delta = np.rollaxis(delta, 2, 1)
    delta = np.rollaxis(delta, 3, 2)
    return delta
# 计算卷积层的反向传播
def conv_backprop(delta,cash):
    delta_c = np.copy(delta)
    delta =swap_first_end_axis(delta)
    a_p = swap_first_end_axis(cash['a_p'])
    jacoby = np.zeros_like(cash['w'])
    for i in range(0,delta.shape[0]):
        for c in range(0,a_p.shape[0]):
            a_p_temp = a_p[np.newaxis,c,:,:,:]
            delta_temp = delta[np.newaxis,i,:,:]
            _,_,_,dw = img2col_conv(a_p_temp,delta_temp,step=1)
            jacoby[i,:,:,c] = dw[0,:,:,0]
    w = cash['w']
    padding_h = w.shape[1] - 1
    padding_w = w.shape[2] - 1
    delta_padding = np.zeros(shape=[delta_c.shape[0],padding_h + delta_c.shape[1] + padding_h,padding_w + delta_c.shape[2] + padding_w,delta_c.shape[3]])
     # 下面要计算前向传播的delta。
    delta_padding[:,padding_h:-padding_h,padding_w:-padding_w] = delta_c
    w = np.flip(w,axis=1)
    w = np.flip(w,axis=2)
    w = swap_first_end_axis(w)
    _, _, _, delta_pre = img2col_conv(delta_padding,w,step=1)

    gradient_dict = {'dw':jacoby,'delta_pre':delta_pre}
    return gradient_dict
def conv_backprop2(delta,cash,converted = True):
    delta_c = np.zeros(shape=[delta.shape[0] * delta.shape[1] * delta.shape[2], delta.shape[3]])
    for i in range(0,delta.shape[0]):
        cursor_start = i * delta.shape[1] * delta.shape[2]
        cursor_end = cursor_start + delta.shape[1] * delta.shape[2]
        for c in range(0,delta.shape[3]):
            unit = delta[i,:,:,c].flatten()
            delta_c[cursor_start:cursor_end,c]=unit
    dw = np.dot(cash['c_a_p'].T,delta_c)
    jacoby = np.zeros_like(cash['w'])
    for i in range(0,dw.shape[1]):
        for c in range(0,jacoby.shape[3]):
            star_p = c * 9
            end_p = star_p + 9
            jacoby[i,:,:,c]= dw[star_p:end_p,i].reshape([jacoby.shape[1],jacoby.shape[2]])
    return {'dw':jacoby}
def tensorHandle(X,shape):
    res=None
    for img in X:
        if res is None:
            res=np.array([img.reshape([*shape])])
        else:
            res=np.concatenate([res,np.array([img.reshape([*shape])])])
    return res

def accuracy(y_predict,y_t):
    return np.mean(np.argmax(y_predict,axis=1)==np.argmax(y_t,axis=1))

if __name__ == '__main__':
    filter1 = np.random.normal(size=[5, 3, 3, 1], loc=0,scale=0.1)
    filter2 = np.random.normal(size=[4, 3, 3, 5], loc=0,scale=0.1)
    w3 = np.random.normal(size=[100, 50], loc=0,scale=0.1)
    w4 = np.random.normal(size=[50, 10], loc=0,scale=0.1)
    paramters = [filter1,filter2,w3,w4]
    train, test = loadMinist()
    x_train,y_train=train
    x_test,y_test=test
    X = x_train
    Y = y_train
    for i in range(0,5000):
        cash,y_p = forward(X=X, Paramters=paramters)
        loss = entrop_loss(y_p, Y)
        if i % 5 == 1:
            _,y_pre = forward(x_test / 255,paramters)
            print("epoch %i , loss:%f  accuracy :%f"%(i,loss,accuracy(y_pre,y_test)))
        delta = y_p - Y
        gradient_dict = full_backprop(delta,cash[-1])
        paramters[3] -= gradient_clip(gradient_dict['dw'] * 0.01,-10,10)
        delta = gradient_dict['delta_pre']
        gradient_dict = full_backprop(delta,cash[-2])
        paramters[2] -= gradient_clip(gradient_dict['dw'] * 0.01, -10, 10)
        delta = gradient_dict['delta_pre']
        delta = pool_backprop(delta,cash[-3])

        gradient_dict = conv_backprop(delta,cash[-4])
        paramters[1] -= gradient_clip((gradient_dict['dw'] / X.shape[0]) * 0.01, -10, 10)
        delta = gradient_dict['delta_pre']
        delta = pool_backprop(delta,cash[-5],flattened=False)

        gradient_dict = conv_backprop(delta,cash[-6])
        paramters[0] -= gradient_clip((gradient_dict['dw'] / X.shape[0]) * 0.01, -10, 10)

