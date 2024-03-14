[TOC]

# 树专题



树结构的重要性不言而喻，以树为中心刷，既能带动链表，又能开启图，算法涉及递归，回溯，动规，贪心，分治等，所以这个结构巨重要。 而树里面二叉树又是重中之重， 所以这篇文章从二叉树的遍历开始 - 复习二叉树的前中后序的递归和非递归代码， 层序遍历之 BFS， 这几个代码非常非常重要。牢牢的掌握这7个框架， 就可以搞掉二叉树这里大部分的题目（大约30道)

**题目1：**[实现二叉树先序，中序和后序遍历](https://www.nowcoder.com/practice/a9fec6c46a684ad5a3abd4e365a9d362?tpId=117&&tqId=35075&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking) 

给定一棵二叉树，分别按照二叉树先序，中序和后序打印所有的节点。

数据范围：0≤n≤1000，树上每个节点的val值满足 0≤val≤100

要求：空间复杂度 O(n)，时间复杂度 O(n)

![img](https://uploadfiles.nowcoder.com/images/20210918/382300087_1631956021286/E61DCE88EB71683589EA3480498477F1)如图二叉树结构

```
示例1
输入：
{1,2,3}
返回值：
[[1,2,3],[2,1,3],[2,3,1]]
```

（1）先序遍历（根左右）

（2）中序遍历（左根右）

（3）后序遍历（左右根）



**题目2：**[给定一个二叉树，返回该二叉树层序遍历的结果，（从左到右，一层一层地遍历）](https://www.nowcoder.com/practice/04a5560e43e24e9db4595865dc9c63a3?tpId=117&&tqId=34936&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

给定一个二叉树，返回该二叉树层序遍历的结果，给定的二叉树是{3,9,20,#,#,15,7},

 <img src="https://uploadfiles.nowcoder.com/images/20210114/999991351_1610616074120/036DC34FF19FB24652AFFEB00A119A76" alt="img" style="zoom:33%;" />

```
 该二叉树层序遍历的结果是
 [
  [3],
  [9,20],
  [15,7] ]
```



## 1、思路框架整理

### 1.1、前序遍历

前中后的递归写法都比较简单，但不能光记这个，非递归也需要会默写，非递归可以更好的帮助我们走遍历过程，另一方面一些变形题目可能涉及到一些中间过程，非递归处理起来会容易些。

#### 递归写法

```
pre_res = []
def PreOrder(root, pre_res):   # 根 和存储前序遍历结果的
	if not root: return

	# 访问或处理根的逻辑
	pre_res.append(root.val)
	
	# 左节点存在，访问
	if root.left:
		PreOrder(root.left, pre_res)
	# 右节点存在，访问
	if root.right:
		PreOrder(root.right, pre_res)
	# 不用返回结果，因为上面的pre_res是全局变量

# 递归前序遍历
PreOrder(root, pre_res)
```

#### 非递归写法

首先既然是递归实现的，我们就需要手动维护个栈， 这样按照前序遍历: 根 -> 左节点 -> 右节点的访问顺序， 当拿到根节点的时候， 需要先访问，然后入栈(为了将来访问右孩子), 然后去他的左孩子， 进行同样的逻辑操作。 当到了最左边没有左孩子的时候， 开始出栈顶部元素，然后去访问右孩子。 
```
pre_res = []
def PreOrder(root, pre_res):
	# 栈来
	stack = []
	# 开始判断
	while root or stack:    # 这个写法比较简单，前中后都可以用
		# 如果当前节点不空， 访问，入栈，找左
		while root:
			# 访问或者处理根的逻辑
			pre_res.append(root.val)
			
			stack.append(pre_res)
			root = root.left
		# 上面循环退出，说明root此时是空，即到了最左边，此时栈顶就是最左边的叶子节点，出栈，访问右孩子
		root = stack.pop()
		root = root.right
	
	# 不用返回结果，因为上面的pre_res是全局变量

# 非递归前序遍历
PreOrder(root, pre_res)
```

### 1.2、中序遍历





