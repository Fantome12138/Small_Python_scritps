import numpy as np

def im2col(im, kernel_size, stride):
    """im2col
    Args:
        im: 输入图像，是一个 shape 为 [c, h, w] 的 np.array。
        kernel_size: 卷积核大小，形式如 [c, k_h, k_w] 
            ，c 为输入 input_data 的通道数。
        stride: stride， 形式如 [h, w]。
    Return:
        out: im2col 结果
    """
    kernel_c, kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    c, h, w = im.shape
    if c != kernel_c:
        raise ValueError("channel: kernel:{}, im:{}".format(kernel_c, c))

    out_h = (h-kernel_h) // stride_h + 1
    out_w = (w-kernel_w) // stride_w + 1
    out = np.zeros((out_h*out_w, kernel_h*kernel_w*kernel_c))
    for idx_h, i in enumerate(range(0, h-kernel_h+1, stride_h)):
        for idx_w, j in enumerate(range(0, w-kernel_w+1, stride_w)):
            out[idx_h*out_w+idx_w, :] = im[:, i:i+kernel_h, j:j+kernel_w].reshape(1, -1)
    return out

def col2im(col, im_size, kernel_size, stride):
    """col2im
    Args:
        col: 输入图像，是一个 shape 为 [num, kernel_w*kernel_h*kernel_c] 的 np.array。
        im_size: 输入时图像的大小，形式如 [c, im_h, im_w]。
        kernel_size: 卷积核大小，形式如 [c, k_h, k_w] 
            ，c 为输入 input_data 的通道数。
        stride: stride， 形式如 [h, w]。
    Return:
        out: col2im 结果
    """
    im_c, im_h, im_w = im_size
    stride_h, stride_w = stride
    kernel_c, kernel_h, kernel_w = kernel_size
    if kernel_c != im_c:
        raise ValueError("channels: kernel:{}, im:{}".format(kernel_c, im_c))

    slice_h = (im_h-kernel_h) // stride_h + 1
    slice_w = (im_w-kernel_w) // stride_w + 1

    out = np.zeros(im_size)
    count = np.zeros(im_size)
    for i in range(slice_h):
        for j in range(slice_w):
            out[:, i*stride_h:i*stride_h+kernel_h, j*stride_w:j*stride_w+kernel_w] = col[i*slice_w+j].reshape(-1, kernel_h, kernel_w)
            count[:, i*stride_h:i*stride_h+kernel_h, j*stride_w:j*stride_w+kernel_w] += 1
    count[count==0] = 1e10
    out /= count
    return out

class BaseLayer():
    
    def __init__(self):
        self._data = dict()
        self._param = dict()

        self._type = "BaseLayer"
        self._name = "NotImplement"

    @property
    def type(self):
        return self._type
    @type.setter
    def type(self, type):
        self._type = type

    def __str__(self):
        return "[{}:{}]".format(self._type, self._name)

    def __call__(self, x):
        return self.forward(x)


    def forward(self, x):
        raise NotImplementedError
    

class Conv2D(BaseLayer):
    """2D 卷积层实现
    包括前向传播与反向传播。
    Attributes:
        __in_channels: 输入图像的通道数。
        __out_channels: 输出图像通道数，也就是卷积核个数。
        __kernel_size: 卷积核大小。
        __stride: tride。
        __padding: padding。
        __bias: 是否 bias。
        _param: 需要的参数，是一个字典，包括 "w" 和 "b" ，在训练阶段还有 "dw" 和 "db" 。
        _data: 输入输出数据，是一个字典，包括 "x" 。
        _type: 层类型。
        _name: 层类名。
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = (self.__in_channels, kernel_size, kernel_size) if isinstance(kernel_size, int) else (self.__in_channels, kernel_size[0], kernel_size[1])
        self.__stride = (stride, stride) if isinstance(stride, int) else stride
        self.__padding = (padding, padding) if isinstance(padding, int) else padding
        self.__bias = bias

        self._param["w"] = np.random.random((self.__kernel_size[0]*self.__kernel_size[1]*self.__kernel_size[2], self.__out_channels))
        self._param["b"] = np.random.random((1, out_channels)) if bias else None

        self._data["x"] = None

        self._name = "Conv2D"
        self._type = "Layer"

        self.__mode = "train"

    def train(self):
        self.__mode = "train"
    def eval(self):
        self.__mode = "eval"
        self._param["dw"] = None
        self._param["db"] = None
        del self._param["dw"]
        del self._param["db"]

    # x: shape [B, c_in,  h, w]
    # y: shape [B, c_out, h, w]
    
    def forward(self, x):
        """前向传播"""
        if self.__mode == "eval":
            self.eval()

        self.__x_c, self.__x_h, self.__x_w = x.shape[1:4]
        if self.__x_c != self.__in_channels:
            raise ValueError("x channels:{}, in_channels:{}".format(self.__x_c, self.__in_channels))

        self._data["x"] = x
        y = []
        self.__x_cols = []
        for x_i in x:
            if self.__padding == (0, 0):
                padding_x = x_i
            else:
                padding_x = np.zeros((self.__x_c, self.__x_h+self.__padding[0]*2, self.__x_w+self.__padding[1]*2))
                padding_x[:, self.__padding[0]:-self.__padding[0], self.__padding[1]:-self.__padding[1]] = x_i
            col, y_i_h, y_i_w = im2col(padding_x, self.__kernel_size, self.__stride)
            self.__x_cols.append(col)
            y_i_single_out_c = np.dot(col, self._param["w"])
            if self.__bias:
                y_i_single_out_c += self._param["b"]
            y_i = []
            for c in range(y_i_single_out_c.shape[1]):
                y_i.append(y_i_single_out_c[:, c].reshape(y_i_h, y_i_w))
            y.append(np.asarray(y_i))
        y = np.asarray(np.asarray(y))
        self.__x_cols = np.asarray(self.__x_cols)
        return y

    def __error2col(self, error):
        b, c, h, w = error.shape
        self.__error_col = np.zeros((b, h*w, c))
        for i in range(b):
            for col in range(c):
                self.__error_col[i, :, col] = error[i, col, :, :].reshape((-1))

    def backward(self, error):
        """反向传播
        # dw = x.T * e
        # db = e
        # dx = e * w.T
        """
        if self.__mode == "eval":
            raise ValueError("mode of {} can't backward".format(self.__mode))

        self._param["dw"] = np.zeros_like(self._param["w"])
        self._param["db"] = np.zeros_like(self._param["b"]) if self.__bias else None

        out_h = (self.__x_h-self.__kernel_size[1]+self.__padding[0]*2) // self.__stride[0] + 1
        out_w = (self.__x_w-self.__kernel_size[2]+self.__padding[0]*2) // self.__stride[1] + 1
        dx = np.zeros((out_h*out_w, self.__kernel_size[0]*self.__kernel_size[1]*self.__kernel_size[2]))

        self.__error2col(error)
        for idx, x_col in enumerate(self.__x_cols):
            self._param["dw"] += np.dot(x_col.T, self.__error_col[idx, :, :])
            dx += np.dot(self.__error_col[idx, :, :], self._param["w"].T)
            if self.__bias:
                self._param["db"] += np.mean(self.__error_col[idx, :, :], axis=0, keepdims=True)

        self._param["dw"] /= (error.shape[0]*out_h*out_w)#*self.__kernel_size[1]**self.__kernel_size[2])
        dx /= error.shape[0]
        if self.__bias:
            self._param["db"] /= (error.shape[0])
        dx = col2im(dx, (self.__x_c, self.__x_h, self.__x_w), self.__kernel_size, self.__stride)
        return dx

    def update(self, lr):
        self._param["w"] -= (lr * self._param["dw"])
        if self.__bias:
            self._param["b"] -= (lr * self._param["db"])

        del self._param["dw"]
        del self._param["db"]


if __name__ == "__main__":
    try:
        import cv2
        import time
        import traceback

        cv2.namedWindow("img_", cv2.WINDOW_NORMAL)
        cv2.namedWindow("out_", cv2.WINDOW_NORMAL)

        img_path = "/Users/fantome/Library/CloudStorage/OneDrive-个人/Git/QR_img/02.jpg"
        kernel, padding, stride = 3, 1, 1
        h, w = 100, 100
        lr = 1e-8

        x = cv2.resize(cv2.imread(img_path), (w, h)).transpose(2, 0, 1)
        c, h, w = x.shape
        print(x.shape)
        x = x[np.newaxis, :, :, :]
        layer_1 = Conv2D(c, 3, kernel, padding, stride)

        loss = 1e10
        while loss > 100:
            y = x

            t = time.time()
            y = layer_1(y)
            layer_1.backward(y-x)
            layer_1.update(lr)
            t = time.time() - t
            loss = np.sum((y-x) ** 2)
            print(loss, t)

            cv2.imshow("img_", x[0].transpose(1, 2, 0))
            out_ = np.asarray((y[0]+0.5).transpose(1, 2, 0), dtype=np.uint8)
            cv2.imshow("out_", out_)
            cv2.waitKey(1)
        print(layer_1.get_w())
        cv2.waitKey(0)
    except:
        traceback.print_exc()

