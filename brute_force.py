"""暴力参照实现：全表扫描。只用于测试对拍和样例校验，不接线上。"""

__all__ = ["BruteForceIndex"]


class BruteForceIndex:
    def __init__(self):
        self._pts = {}  # bid -> (lat, lon)

    def __len__(self):
        return len(self._pts)

    def insert(self, bid, lat, lon):
        if bid in self._pts:
            raise ValueError("编号已存在: %d" % bid)
        self._pts[bid] = (lat, lon)

    def remove(self, bid):
        if bid not in self._pts:
            raise ValueError("编号不存在: %r" % (bid,))
        del self._pts[bid]

    def update(self, bid, lat, lon):
        if bid not in self._pts:
            raise ValueError("编号不存在: %r" % (bid,))
        self._pts[bid] = (lat, lon)

    def rect(self, min_lat, min_lon, max_lat, max_lon):
        if min_lat > max_lat or min_lon > max_lon:
            return []
        return sorted(
            bid for bid, (lat, lon) in self._pts.items()
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
        )

    def nearest(self, lat, lon, k):
        if k <= 0:
            return []
        scored = (
            ((lat - la) * (lat - la) + (lon - lo) * (lon - lo), bid)
            for bid, (la, lo) in self._pts.items()
        )
        return [bid for _, bid in sorted(scored)[:k]]
