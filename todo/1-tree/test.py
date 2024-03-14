pre_res = [1,2,3]

class Root:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def PerOrder(root, pre_res):
    if not root: return 0
    pre_res.append(root.val)
    if root.left:
        PerOrder(root.left, pre_res)
    if root.right:
        PerOrder(root.right, pre_res)

root = Root
PerOrder(root, pre_res)




