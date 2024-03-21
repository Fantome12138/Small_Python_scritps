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

def inOrder(root, pre_res):
    if not root:
        return 0
    if root.left:
        inOrder(root.left, pre_res)
    pre_res.append(root.val)
    if root.right:
        inOrder(root.right, pre_res)
    return pre_res

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

def postOrder(root, pre_res):
    if not root:
        return 0
    if root.right:
        postOrder(root.right, pre_res)
    pre_res.append(root.val)
    if root.left:
        postOrder(root.left, pre_res)
    return pre_res

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


if __name__ == "__main__":
    pre_res = []
    root = Tree(1)
    root.left = Tree(2)
    root.right = Tree(3)
    _res = postOrder_recursive(root, pre_res)
    print(_res)
