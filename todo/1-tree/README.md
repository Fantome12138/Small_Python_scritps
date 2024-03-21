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

### 1.1 前序遍历

前中后的递归写法都比较简单，但不能光记这个，非递归也需要会默写，非递归可以更好的帮助我们走遍历过程，另一方面一些变形题目可能涉及到一些中间过程，非递归处理起来会容易些。

#### 递归写法

```
class Tree():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
 
def PerOrder(root, pre_res):
    if not root:
        return 0
    pre_res.append(root.val)
    if root.left:
        PerOrder(root.left, pre_res)
    if root.right:
        PerOrder(root.right, pre_res)
    return pre_res
```

#### 非递归写法

首先既然是递归实现的，就需要手动维护个栈， 这样按照前序遍历: 根 -> 左节点 -> 右节点的访问顺序， 当拿到根节点的时候， 需要先访问，然后入栈(为了将来访问右孩子), 然后去他的左孩子， 进行同样的逻辑操作。 当到了最左边没有左孩子的时候， 开始出栈顶部元素，然后去访问右孩子。 
```
def PerOrder_recursive(root, pre_res): # 非递归
    stack = []
    while root or stack:
        while root:
            pre_res.append(root.val)
            stack.append(root)
            root = root.left
        root = stack.pop()
        root = root.right
    return pre_res
```

### 1.2 中序遍历

#### 递归写法

```
def inOrder(root, pre_res):
    if not root:
        return 0
    if root.left:
        inOrder(root.left, pre_res)
    pre_res.append(root.val)
    if root.right:
        inOrder(root.right, pre_res)
    return pre_res
```

#### 非递归写法

- 非递归手动维护栈
- 左孩子 -> 根 -> 右孩子
- 当我们拿到root， 我们要先入栈，然后找左孩子，到最左边之后，访问，然后去他的右孩子

```
def inOrder_recursive(root, pre_res): # 非递归
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        pre_res.append(root.val)
        root = root.right
    return pre_res
```

### 1.3 后序遍历

#### 递归写法

```
def postOrder(root, pre_res):
    if not root:
        return 0
    if root.right:
        postOrder(root.right, pre_res)
    pre_res.append(root.val)
    if root.left:
        postOrder(root.left, pre_res)
    return pre_res
```

#### 非递归写法

- 手动维护栈
- 左孩子 -> 右孩子 -> 根
- 这里需要判断当前root是从他左孩子还是右孩子处返回来的，如果是左孩子返回来的，需要去访问右孩子，如果是右孩子返回来的， 需要访问root
- 这里的一个处理技巧就是往左找的时候，我们不是找最左边那个了，而是找第一个需要访问的叶子节点， 有了第一个节点，后面就好说了，先访问这个节点，然后判断它是它父亲的左孩子还是右孩子（它父亲在栈里面），如果是左孩子， 那么去访问它父亲的右孩子， 如果是右孩子， 当前置为空，下一轮直接访问它父亲正好

```
def postOrder_recursive(root, pre_res):
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left if root.left else root.right
            # 这样root为空的时候，栈顶就不是最左边了，而是第一个应该访问的叶子，出栈访问它
        root = stack.pop()
        pre_res.append(root.val)
        # 下面判断是它父亲的左节点还是右节点
        if stack and stack[-1].left == root:   # 是它父亲的左孩子、
            root = stack[-1].right    # 直接访问它父亲右孩子
        else:
            root = None   # 否则，说明这个分支完事了，下轮直接访问它父亲即可
    return pre_res
```

### 1.4 层序遍历

树的层序遍历属于bfs的范畴了，bfs两个重大应用场景：层序遍历和最短路径。

这里首先定式， 提到**bfs先想到队列， dfs先想到栈**。关于层次遍历是怎么回事，这里不解释， 这里整理两个模板：

baseline模板， 这个就是传统的bfs遍历模板，巨重要：

```
import collections
from collections import deque
def bfs(root):
    if not root:
        return 0
    d = collections.deque([])
    d.append(root)
    while d:
        root = d.popleft()
        if root.left:
            d.append(root.left)
        if root.right:
            d.append(root.right)
```

但这个无法区分队列中的节点来自哪一层。所以这个还不能直接拿来做二叉树的层序遍历。

层序遍历要求的输入结果和 BFS 是不同的。层序遍历要求我们区分每一层，也就是返回一个二维数组。而 BFS 的遍历结果是一个一维数组，无法区分每一层。

所以层序遍历的话， 我们需要在每一层遍历开始前，先记录队列中的结点数量 n nn（也就是这一层的结点数量），然后一口气处理完这一层的n 个结点。上代码：

```
def levelOrder(root, pre_res):
    if not root:
        return 0
    d = deque([root])
    while d:
        print(len(d))
        level_tmp = []
        for _ in range(len(d)):
            root = d.popleft()
            level_tmp.append(root.val)
            if root.left:
                d.append(root.left)
            if root.right:
                d.append(root.right)
        pre_res.append(level_tmp)
    return pre_res
```

## 2、题目思路和代码整理

#### 











