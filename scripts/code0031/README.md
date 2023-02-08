#### 特征提取、筛选、匹配【代码见some_python_scripts/code0031】

**使用opencv提取图像特征：AKAZE、ORB**



**使用opencv对提取的特征进行筛选、匹配：**

OpenCV里面的二维特征匹配，有两种常用的方法，

- Brute Force Matcher
- Flann based Matcher

继承于DescriptorMatcher类，分别对应BFMatcher和FlannBasedMatcher。

二者的区别： BFMatcher总是尝试所有可能的匹配，从而使得它总能够找到最佳匹配。这也是Brute Force（暴力法）的原始含义。方法是计算某一个特征点描述子与其他所有特征点描述子之间的距离，然后将得到的距离进行排序，取距离最近的一个作为匹配点。 FlannBasedMatcher中FLANN的含义是Fast Library for Approximate Nearest Neighbors，它是一种近似法，算法更快但是找到的是最近邻近似匹配，当我们需要找到一个相对好的匹配但是不需要最佳匹配的时候可以用FlannBasedMatcher。当然也可以通过调整FlannBasedMatcher的参数来提高匹配的精度或者提高算法速度，但是相应地算法速度或者算法精度会受到影响。

此外，使用特征提取过程得到的特征描述符（descriptor）数据类型有的是float类型的，ORB和BRIEF特征描述子只能使用BruteForce匹配法。

具体matching的方法

- match: 给定查询集合中的每个特征描述子，寻找最佳匹配，返回值按距离升序排列。
- knnMatch：给定查询集合中的每个特征描述子，寻找k个最佳匹配
- radiusMatch：在特定范围内寻找特征描述子，返回特定距离内的匹配

###### 特征点筛选

通过上述过程得到了raw_matches之后，接下来对其进行筛选。

- Cross-match filter

这种方法只针对BFMatcher, 就是将BFMatcher的最后一个参数，交叉验证声明为true，即matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)，如果图1的一个特征点和图2的一个特征点相匹配，则进行一个相反的检查，即从图2上的特征点进行匹配图1的特征点，如果相互之间都匹配成功，才认为是一个好的匹配。

- knnMatch

knnMatch返回K个好的匹配，k可以自行指定。这里指定k=2，raw_matches = matcher.knnMatch(desc1, desc2,2) ，然后每个match得到两个最接近的descriptor，再计算最接近距离和次接近距离之间的比值，当比值大于某个设定的值时，才作为最终的match。

- RANSAC （最好的）

为了进一步提升精度，还可以采用随机采样一致性（RANSAC）来过滤错误的匹配，该方法是利用匹配点计算两图像之间的单应矩阵，然后利用重投影误差来判定某一个匹配是否是正确的匹配。OpenCV中封装了求解单应矩阵的方法 findHomography ,可以为该方法设定一个重投影误差的阈值，可以得到一个向量mask来指定那些是符合重投影误差的匹配点对，用来过滤掉错误的匹配。