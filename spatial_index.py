"""共享单车空间索引库。

二维 kd-tree + scapegoat 动态平衡，全程整数运算（微度坐标），
支持 insert / remove / update / rect / nearest 五种操作。

- insert / remove / update：均摊 O(log n)
- rect：O(log n + m)，m 为命中数
- nearest：O(log n + k log k)

删除采用墓碑标记，比例过高时整体重建；插入沿路径检查平衡因子，
失衡时对子树重建（scapegoat），保证任意增删序列下树高保持 O(log n)，
不会越用越慢，也不残留垃圾节点。
"""

from heapq import heappush, heapreplace

__all__ = ["SpatialIndex"]

LAT_MIN, LAT_MAX = -90_000_000, 90_000_000
LON_MIN, LON_MAX = -180_000_000, 180_000_000

# 平衡因子 alpha = 3/4：任一子树超过父节点 3/4 的重量即重建
_ALPHA_NUM = 3
_ALPHA_DEN = 4


class _Node:
    __slots__ = ("bid", "lat", "lon", "axis", "left", "right", "total", "alive")

    def __init__(self, bid, lat, lon, axis):
        self.bid = bid
        self.lat = lat
        self.lon = lon
        self.axis = axis
        self.left = None
        self.right = None
        self.total = 1      # 子树结构节点数（含墓碑），用于平衡判断
        self.alive = True    # 删除后置 False（墓碑）


def _check_coords(lat, lon):
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        raise ValueError("坐标越界: lat=%r lon=%r" % (lat, lon))


class SpatialIndex:
    """车辆位置的空间索引。坐标为整数微度，车辆编号为正整数。"""

    def __init__(self):
        self._root = None
        self._nodes = {}   # bid -> _Node，O(1) 定位
        self._live = 0     # 存活车辆数
        self._dead = 0     # 墓碑节点数

    def __len__(self):
        return self._live

    # ---------------------------------------------------------------- 增删改

    def insert(self, bid, lat, lon):
        """插入一辆车。编号已存在时抛 ValueError。"""
        if not isinstance(bid, int) or bid <= 0:
            raise ValueError("编号必须是正整数: %r" % (bid,))
        _check_coords(lat, lon)
        if bid in self._nodes:
            raise ValueError("编号已存在: %d" % bid)
        node = _Node(bid, lat, lon, 0)
        self._nodes[bid] = node
        self._live += 1
        if self._root is None:
            self._root = node
            return
        path = []
        cur = self._root
        while True:
            path.append(cur)
            cur.total += 1
            if self._less(node, cur):
                nxt = cur.left
                if nxt is None:
                    node.axis = cur.axis ^ 1
                    cur.left = node
                    break
            else:
                nxt = cur.right
                if nxt is None:
                    node.axis = cur.axis ^ 1
                    cur.right = node
                    break
            cur = nxt
        # scapegoat：沿路径找最浅的失衡祖先，重建其子树
        for i in range(len(path)):
            anc = path[i]
            child = path[i + 1] if i + 1 < len(path) else node
            if child.total * _ALPHA_DEN > anc.total * _ALPHA_NUM:
                new_sub = self._rebuild(anc)
                if i == 0:
                    self._root = new_sub
                else:
                    parent = path[i - 1]
                    if parent.left is anc:
                        parent.left = new_sub
                    else:
                        parent.right = new_sub
                # 重建丢弃了墓碑，沿路径向上修正祖先的子树重量
                for j in range(i - 1, -1, -1):
                    n = path[j]
                    n.total = 1 + (n.left.total if n.left else 0) \
                              + (n.right.total if n.right else 0)
                break

    def remove(self, bid):
        """删除一辆车。编号不存在时抛 ValueError。"""
        node = self._nodes.pop(bid, None)
        if node is None:
            raise ValueError("编号不存在: %r" % (bid,))
        node.alive = False
        self._live -= 1
        self._dead += 1
        # 墓碑达到存活数时整体重建，回收内存并恢复平衡
        if self._dead >= self._live and self._dead > 0:
            self._root = self._rebuild(self._root)

    def update(self, bid, lat, lon):
        """把车移到新位置，等价于删掉再插入。编号不存在时抛 ValueError。"""
        _check_coords(lat, lon)
        if bid not in self._nodes:
            raise ValueError("编号不存在: %r" % (bid,))
        self.remove(bid)
        self.insert(bid, lat, lon)

    @staticmethod
    def _less(a, b):
        """按 b.axis 维度比较，并列时依次用另一维、编号定序，保证全序。"""
        if b.axis == 0:
            return (a.lat, a.lon, a.bid) < (b.lat, b.lon, b.bid)
        return (a.lon, a.lat, a.bid) < (b.lon, b.lat, b.bid)

    def _rebuild(self, root):
        """把子树重建成完全平衡的 kd-tree，丢弃墓碑节点。"""
        if root is None:
            return None
        nodes = []
        stack = [root]
        dead = 0
        while stack:
            n = stack.pop()
            if n.alive:
                nodes.append(n)
            else:
                dead += 1
            if n.left is not None:
                stack.append(n.left)
            if n.right is not None:
                stack.append(n.right)
        self._dead -= dead
        return self._build(nodes, 0)

    def _build(self, nodes, depth):
        if not nodes:
            return None
        axis = depth & 1
        if axis == 0:
            nodes.sort(key=lambda n: (n.lat, n.lon, n.bid))
        else:
            nodes.sort(key=lambda n: (n.lon, n.lat, n.bid))
        mid = len(nodes) // 2
        root = nodes[mid]
        root.axis = axis
        root.total = len(nodes)
        root.left = self._build(nodes[:mid], depth + 1)
        root.right = self._build(nodes[mid + 1:], depth + 1)
        return root

    # ---------------------------------------------------------------- 查询

    def rect(self, min_lat, min_lon, max_lat, max_lon):
        """闭区间矩形查询，返回编号升序列表。反向矩形返回空列表。"""
        if min_lat > max_lat or min_lon > max_lon:
            return []
        out = []
        self._rect(self._root, min_lat, min_lon, max_lat, max_lon, out)
        out.sort()
        return out

    def _rect(self, node, min_lat, min_lon, max_lat, max_lon, out):
        if node is None:
            return
        if node.axis == 0:
            v, lo, hi = node.lat, min_lat, max_lat
            inside = lo <= v <= hi and min_lon <= node.lon <= max_lon
        else:
            v, lo, hi = node.lon, min_lon, max_lon
            inside = lo <= v <= hi and min_lat <= node.lat <= max_lat
        if v >= lo:
            self._rect(node.left, min_lat, min_lon, max_lat, max_lon, out)
        if node.alive and inside:
            out.append(node.bid)
        if v <= hi:
            self._rect(node.right, min_lat, min_lon, max_lat, max_lon, out)

    def nearest(self, lat, lon, k):
        """返回距 (lat, lon) 最近的 k 辆，按 (距离平方, 编号) 升序。"""
        if k <= 0:
            return []
        heap = []  # 最大堆（存相反数）：堆顶是当前第 k 远的结果
        self._nearest(self._root, lat, lon, k, heap)
        return [bid for _, bid in sorted((-d, -b) for d, b in heap)]

    def _nearest(self, node, lat, lon, k, heap):
        if node is None:
            return
        if node.axis == 0:
            diff = lat - node.lat
        else:
            diff = lon - node.lon
        if diff < 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left
        self._nearest(near, lat, lon, k, heap)
        if node.alive:
            dlat = lat - node.lat
            dlon = lon - node.lon
            d2 = dlat * dlat + dlon * dlon
            entry = (-d2, -node.bid)
            if len(heap) < k:
                heappush(heap, entry)
            elif entry > heap[0]:
                heapreplace(heap, entry)
        # 剪枝：分割面距离平方已超过当前第 k 远，远侧不可能更优。
        # 等号不能剪：距离并列时还要靠编号定序。
        if len(heap) < k or diff * diff <= -heap[0][0]:
            self._nearest(far, lat, lon, k, heap)

    # ---------------------------------------------------------------- 自检

    def stats(self):
        """返回 (存活数, 墓碑数, 结构节点总数)，供测试与监控用。"""
        total = self._live + self._dead
        return (self._live, self._dead, total)
