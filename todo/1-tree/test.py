pre_res = []

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def PerOrder(root, pre_res):
    if not root: return 0
    pre_res.append(root.val)
    if root.left:
        PerOrder(root.left, pre_res)
    if root.right:
        PerOrder(root.right, pre_res)
    return pre_res


root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
_res = PerOrder(root, pre_res)
print(_res)
