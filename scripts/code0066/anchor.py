import numpy as np
import time
# size:在原图上生成区域的大小[32,64,128..]
# ratio:对应宽高/高宽比，其实差别不大[0.5,1,2]
# scale:为对应某个块大小的面积扩增/缩减的比例,[0.7,1,1.5]
#针对某个尺寸 32,生成 [22.4,32,48]的尺寸
# fea_size:所对应阶段特征图的尺寸[64,32,16]
# strides:[8,16,32]
# names:自定义的名字列表
# root：存储数据的文件夹
def getAnchor(size,ratio,scale,fea_size,strides,names,root):
	num_base_anchors=len(ratio)*len(scale) 
	result=[]
	start=time.time()
	for index,s in enumerate(size):
		#每次需要置0
		base_anchors=np.zeros((num_base_anchors,4),dtype=np.float32)
		#np.tile(scale,(2,len(ratio)))scale[1,3]沿y轴扩增2倍，横轴扩增3倍
		#则最后变成了2*9 需要转置
		base_anchors[:,2:]=s*np.tile(scale,(2,len(ratio))).T 
		areas = base_anchors[:, 2] * base_anchors[:, 3]#求每个anchors的面积
		# 在scale的基础上进行ratio的计算 
		# 形成经过scale和ratio 组合后的 9 个坐标 
		# 但此时这个坐标仍然是后两维是宽和长（取决于ratio为长高比还是宽高比）
		# 前两维仍然是0，所以需要以当前位置为中心将其换为(x1,y1,x2,y2)的形式
		base_anchors[:,2]=np.sqrt(areas / np.repeat(ratio,len(scale)))
		base_anchors[:,3]=base_anchors[:,2]*np.repeat(ratio,len(scale))
		# 以本位置为中心 换为(x1,y1,x2,y2)的形式
		base_anchors[:,0]-=0.5*base_anchors[:,2]
		base_anchors[:,1]-=0.5*base_anchors[:,3]
		base_anchors[:,2]-=0.5*base_anchors[:,2]
		base_anchors[:,3]-=0.5*base_anchors[:,3]
		#cur_result=[]#用来存储当前尺寸下的变换结果
		# 此时将其映射到原图，特征图的大小假设是怡512*512为输入，
		# 则在3-7层的特征图输出尺寸为fea_size那个数组，这里需要逐次遍历这个数组
		# 这个需要对数组进行间隔取值，比如64*64大小的，其实就是将原图512*512划分成了
		# 64*64大小的块，每个块的中心坐标加上我们以上base_anchors的偏移量即为我们的所求 
		# 如何进行分块组合 这个需要用到 np.meshgrid 函数
		#for fea,stride in zip([fea_size[index]],[strides[index]]):
		#此步得到将512分块后每个块(x,y)的中心位置（上取整）
		shift_x=(np.arange(0,fea_size[index][0])+0.5)*strides[index]
		shift_y=(np.arange(0,fea_size[index][1])+0.5)*strides[index]
		# 使用meshgrid函数进行广播数组
		# 此时为shift_x为其每一行重复原来的shift_x(行向量 1*fea[0])
		# 此时为shift_y为其每一列重复原来的shift_y(行向量 1*fea[1])
		shift_x,shift_y=np.meshgrid(shift_x,shift_y)
		#下一步进行合并shift_x shift_y，即将其组合为先x轴按步长累加，
		# 然后将按列按步长进行累加
		shift_x=np.reshape(shift_x,[-1])#展为行向量(1,fea[1]*fea[0])
		shift_y=np.reshape(shift_y,[-1])#展为行向量(1,fea[0]*fea[1])
		# 每个位置为 x y的中心位置，
		# 再加上之前的那个在以自身为中心偏移过的base_anchors 
		# 即为特征图上某点到原图的映射
		# shifts:4*(fea[0]*fea[1])
		shifts=np.stack([shift_x,shift_y,shift_x,shift_y],axis=0)
		shifts=np.transpose(shifts)# 转置即为所求
		# 将原图的对应中心和base_anchors进行加和组装
		number_of_anchors = np.shape(base_anchors)[0]
		k = np.shape(shifts)[0]
		#将9个anchors分别和当前块的中心点相加
		shifted_anchors = np.reshape(base_anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), np.float32)
		shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])#reshape为所求
		#cur_result.append(shifted_anchors)#将其放进当前结果集中
		result.append(shifted_anchors)#将该尺寸的所有框放在
  
	print("time:",time.time()-start," len ",sum([i.shape[0] for i in result]))
	return result # list (all_level_num_anchors,4)