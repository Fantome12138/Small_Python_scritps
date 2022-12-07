import functools
import time
from _thread import RLock


def make_key(args, kwds):
    """
    通过方法参数获取缓存的 key
    :param args: 方法原始参数
    :param kwds: 方法键值对参数
    :return: 计算后 key 的 hash 值
    """
    key = args
    if kwds:
        key += object()
        print('1', object(), type(object()))
        for item in kwds.items():
            key += item
    elif len(key) == 1 and type(key[0]) in {int, str, frozenset, type(None)}:
        return key[0]
    return hash(key)


def lru_cache(maxsize=128):
    def decorating_function(user_function):
        PREV, NEXT, KEY, RESULT = 0, 1, 2, 3

        cache = {}                          # 缓存结构，存储 key 与缓存数据的映射
        hits = misses = 0                   # 缓存命中、丢失统计
        full = False                        # 缓冲区是否已满标记
        cache_len = cache.__len__           # 缓冲区大小
        root = []
        root[:] = [root, root, None, None]  # 队列头结点为空

        def lru_cache_wrapper(*args, **kwds):
            nonlocal root, hits, misses, full
            key = make_key(args, kwds)
            with RLock():  # 线程安全锁
                link = cache.get(key)

                """ 缓存命中，移动命中节点，直接返回预设结果 """
                if link is not None:
                    """ 从链表中移除命中节点 """
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev

                    """ 将命中节点移动到队尾 """
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result

            """ 缓存未命中，调用方法获取返回，创建节点，淘汰算法 """
            result = user_function(*args, **kwds)
            with RLock():
                if key in cache:
                    pass
                elif full:
                    """ 缓冲区已满，淘汰队首元素，删除缓冲区对应元素 """
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    root[KEY] = root[RESULT] = None
                    del cache[oldkey]
                    cache[key] = oldroot
                else:
                    """ 缓冲区未满，直接创建节点，插入数据 """
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    full = (cache_len() >= maxsize)
                misses += 1
            return result
        return lru_cache_wrapper

    return decorating_function


def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r ' % (elapsed, name, arg_str, result))
        return result

    return clocked


@lru_cache()
@clock
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 2) + fibonacci(n - 1)


if __name__ == '__main__':
    print(fibonacci(10))


