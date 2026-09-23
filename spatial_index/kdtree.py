"""动态 k-d 树空间索引（2 维，整数坐标，纯标准库）。

坐标单位是微度（经纬度乘 1e6 取整），距离用整数平方欧氏距离，
全程无浮点运算，排序结果与机器无关。

结构说明：
- 树节点存放在平行的 array('q') 里（_lat/_lon/_pid/_left/_right/_size），
  节点之间用下标引用，-1 表示空子树；定长数组避免每个节点一堆
  Python 对象头，50 万个点常驻内存远低于 200 MB。
- 插入沿树下行，按层交替比较维度；发现某棵子树失衡
  （较大子树 > 3/4 整棵树）时，把该子树收集起来重建成完全平衡树
  （scapegoat 策略），保证均摊 O(log n) 插入。
- 删除是惰性删除：节点打上墓碑标记，结构不变；当墓碑数超过存活数时
  整树重建一次，回收全部墓碑节点。因此频繁增删不会越用越慢，
  内存占用也只跟当前存活点数成正比。
- 被重建丢弃的节点槽位进入空闲链表复用，不会只增不减。
- 查询不修改任何内部状态，查多少次索引占用都不变。
"""

from array import array
from heapq import heappush, heapreplace
from operator import itemgetter

LAT_MIN, LAT_MAX = -90000000, 90000000
LON_MIN, LON_MAX = -180000000, 180000000

# 子树重构阈值：较大子树 * _NUM > 整树 * _DEN 时重建
_REBAL_NUM = 4
_REBAL_DEN = 3


def _check_coords(lat_e6, lon_e6):
    if not (LAT_MIN <= lat_e6 <= LAT_MAX and LON_MIN <= lon_e6 <= LON_MAX):
        raise ValueError(
            "coordinate out of range: lat_e6=%r lon_e6=%r" % (lat_e6, lon_e6)
        )


class KDTreeIndex:
    """支持动态增删改的二维空间索引。

    接口：
        insert(id, lat_e6, lon_e6)  编号已存在抛 KeyError
        remove(id)                  编号不存在抛 KeyError
        update(id, lat_e6, lon_e6)  等价于 remove + insert
        rect(min_lat, min_lon, max_lat, max_lon) -> [id, ...] 按编号升序
        nearest(lat_e6, lon_e6, k) -> [id, ...] 按 (距离平方, 编号) 升序
    """

    __slots__ = (
        "_lat", "_lon", "_pid", "_left", "_right", "_size", "_alive",
        "_index", "_free", "_root", "_live", "_dead",
    )

    def __init__(self, points=None):
        self._lat = array("q")    # 节点纬度（微度）
        self._lon = array("q")    # 节点经度（微度）
        self._pid = array("q")    # 节点车辆编号
        self._left = array("q")   # 左子树下标，-1 为空
        self._right = array("q")  # 右子树下标，-1 为空
        self._size = array("q")   # 子树结构大小（含墓碑，仅供平衡判断）
        self._alive = bytearray()  # 1 存活，0 墓碑
        self._index = {}   # id -> 节点下标
        self._free = []    # 可回收的节点槽位
        self._root = -1
        self._live = 0    # 存活点数
        self._dead = 0    # 墓碑数
        if points:
            pts = [(lat, lon, pid) for pid, lat, lon in points]
            ids = self._index
            for _, _, pid in pts:
                if pid in ids:
                    raise KeyError("duplicate id: %r" % (pid,))
                ids[pid] = 0  # 占位，真正下标在建树时写
            self._root = self._build(pts, 0)
            self._live = len(pts)

    # ------------------------------------------------------------------
    # 基本信息
    # ------------------------------------------------------------------
    def __len__(self):
        return self._live

    def __contains__(self, pid):
        return pid in self._index

    # ------------------------------------------------------------------
    # 建树 / 重构
    # ------------------------------------------------------------------
    def _alloc(self, lat, lon, pid):
        """分配一个节点槽位，优先复用空闲槽。"""
        if self._free:
            slot = self._free.pop()
            self._lat[slot] = lat
            self._lon[slot] = lon
            self._pid[slot] = pid
            self._left[slot] = -1
            self._right[slot] = -1
            self._size[slot] = 1
            self._alive[slot] = 1
            return slot
        self._lat.append(lat)
        self._lon.append(lon)
        self._pid.append(pid)
        self._left.append(-1)
        self._right.append(-1)
        self._size.append(1)
        self._alive.append(1)
        return len(self._pid) - 1

    def _build(self, pts, depth):
        """把 [(lat, lon, pid), ...] 重建成平衡子树，返回根下标。

        每层按当前维度取中位数分裂，递归深度 O(log n)。
        """
        if not pts:
            return -1
        axis = depth & 1
        pts.sort(key=itemgetter(axis))
        mid = len(pts) >> 1
        lat, lon, pid = pts[mid]
        slot = self._alloc(lat, lon, pid)
        self._index[pid] = slot
        left = self._build(pts[:mid], depth + 1)
        right = self._build(pts[mid + 1:], depth + 1)
        self._left[slot] = left
        self._right[slot] = right
        self._size[slot] = len(pts)
        return slot

    def _recycle(self, root, pts):
        """收集子树里的存活点，并把全部旧槽位放进空闲链。"""
        stack = [root]
        lat_a, lon_a, pid_a = self._lat, self._lon, self._pid
        left_a, right_a, alive_a = self._left, self._right, self._alive
        free = self._free
        while stack:
            s = stack.pop()
            if alive_a[s]:
                pts.append((lat_a[s], lon_a[s], pid_a[s]))
            free.append(s)
            l = left_a[s]
            if l != -1:
                stack.append(l)
            r = right_a[s]
            if r != -1:
                stack.append(r)

    def _rebuild_subtree(self, root, depth):
        pts = []
        self._recycle(root, pts)
        self._dead -= self._size[root] - len(pts)
        return self._build(pts, depth)

    # ------------------------------------------------------------------
    # 增删改
    # ------------------------------------------------------------------
    def insert(self, pid, lat_e6, lon_e6):
        _check_coords(lat_e6, lon_e6)
        index = self._index
        if pid in index:
            raise KeyError("duplicate id: %r" % (pid,))
        if self._root == -1:
            self._root = self._alloc(lat_e6, lon_e6, pid)
            index[pid] = self._root
            self._live += 1
            return

        lat_a, lon_a, pid_a = self._lat, self._lon, self._pid
        left_a, right_a, size_a = self._left, self._right, self._size
        node = self._root
        depth = 0
        path = []
        while True:
            path.append(node)
            if depth & 1:
                go_left = (lon_e6, lat_e6, pid) < (lon_a[node], lat_a[node], pid_a[node])
            else:
                go_left = (lat_e6, lon_e6, pid) < (lat_a[node], lon_a[node], pid_a[node])
            nxt = left_a[node] if go_left else right_a[node]
            if nxt == -1:
                slot = self._alloc(lat_e6, lon_e6, pid)
                if go_left:
                    left_a[node] = slot
                else:
                    right_a[node] = slot
                index[pid] = slot
                break
            node = nxt
            depth += 1

        self._live += 1
        # 更新路径上的子树大小，并找出最高的失衡祖先（scapegoat）
        for ancestor in path:
            size_a[ancestor] += 1
        scapegoat = -1
        scapegoat_depth = 0
        for i, ancestor in enumerate(path):
            s = size_a[ancestor]
            l = left_a[ancestor]
            r = right_a[ancestor]
            big = size_a[l] if l != -1 else 0
            if r != -1 and size_a[r] > big:
                big = size_a[r]
            if big * _REBAL_DEN > s * _REBAL_NUM:
                scapegoat = ancestor
                scapegoat_depth = i
        if scapegoat != -1:
            new_root = self._rebuild_subtree(scapegoat, scapegoat_depth)
            if scapegoat_depth == 0:
                self._root = new_root
            else:
                parent = path[scapegoat_depth - 1]
                if left_a[parent] == scapegoat:
                    left_a[parent] = new_root
                else:
                    right_a[parent] = new_root
                # 重构后子树大小可能变小（墓碑被回收），
                # 重新统计父链上的 _size，保持平衡判断可靠
                for j in range(scapegoat_depth - 1, -1, -1):
                    p = path[j]
                    l = left_a[p]
                    r = right_a[p]
                    total = 1
                    if l != -1:
                        total += size_a[l]
                    if r != -1:
                        total += size_a[r]
                    size_a[p] = total

    def remove(self, pid):
        slot = self._index.pop(pid, None)
        if slot is None:
            raise KeyError("unknown id: %r" % (pid,))
        self._alive[slot] = 0
        self._live -= 1
        self._dead += 1
        # 墓碑超过存活数时整树重建，回收全部墓碑，
        # 保证树高和内存都只跟存活点数相关
        if self._dead > self._live and self._root != -1:
            self._root = self._rebuild_subtree(self._root, 0)

    def update(self, pid, lat_e6, lon_e6):
        _check_coords(lat_e6, lon_e6)
        if pid not in self._index:
            raise KeyError("unknown id: %r" % (pid,))
        self.remove(pid)
        self.insert(pid, lat_e6, lon_e6)

    # ------------------------------------------------------------------
    # 查询（只读，不改变任何内部状态）
    # ------------------------------------------------------------------
    def rect(self, min_lat, min_lon, max_lat, max_lon):
        """闭区间矩形查询，结果按编号升序。反向矩形返回空列表。"""
        if min_lat > max_lat or min_lon > max_lon or self._root == -1:
            return []
        lat_a, lon_a, pid_a = self._lat, self._lon, self._pid
        left_a, right_a, alive_a = self._left, self._right, self._alive
        hits = []
        stack = [(self._root, 0)]
        while stack:
            node, axis = stack.pop()
            lat = lat_a[node]
            lon = lon_a[node]
            if alive_a[node] and min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                hits.append(pid_a[node])
            if axis:
                if min_lon <= lon and left_a[node] != -1:
                    stack.append((left_a[node], 0))
                if lon <= max_lon and right_a[node] != -1:
                    stack.append((right_a[node], 0))
            else:
                if min_lat <= lat and left_a[node] != -1:
                    stack.append((left_a[node], 1))
                if lat <= max_lat and right_a[node] != -1:
                    stack.append((right_a[node], 1))
        hits.sort()
        return hits

    def nearest(self, lat_e6, lon_e6, k):
        """最近的 k 辆，按 (距离平方, 编号) 升序；k <= 0 返回空。"""
        if k <= 0 or self._root == -1:
            return []
        lat_a, lon_a, pid_a = self._lat, self._lon, self._pid
        left_a, right_a, alive_a = self._left, self._right, self._alive
        # 堆元素 (-dist2, -pid)：堆顶是当前第 k 远（最差）的点
        heap = []
        # 栈元素 (节点, 维度, 下界)：下界是该子树所有点到查询点的最小可能
        # 距离平方（近侧子树为 0，远侧子树为分割面距离平方）。弹出时若
        # 下界已大于当前第 k 远的距离平方，整棵子树剪掉。
        stack = [(self._root, 0, 0)]
        while stack:
            node, axis, bound = stack.pop()
            if heap and len(heap) >= k and bound > -heap[0][0]:
                continue
            dlat = lat_a[node] - lat_e6
            dlon = lon_a[node] - lon_e6
            if alive_a[node]:
                item = (-(dlat * dlat + dlon * dlon), -pid_a[node])
                if len(heap) < k:
                    heappush(heap, item)
                elif item > heap[0]:
                    heapreplace(heap, item)
            # 近侧（查询点所在一侧）后压栈、先展开，让第 k 远的距离
            # 尽快收紧；远侧带上下界，留到弹出时再判断是否剪枝
            if axis:
                diff = dlon
            else:
                diff = dlat
            if diff < 0:
                near = right_a[node]
                far = left_a[node]
            else:
                near = left_a[node]
                far = right_a[node]
            if far != -1:
                stack.append((far, axis ^ 1, diff * diff))
            if near != -1:
                stack.append((near, axis ^ 1, 0))
        result = sorted((-neg_d, -neg_p) for neg_d, neg_p in heap)
        return [pid for _, pid in result]
