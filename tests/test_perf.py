"""性能与规模测试：50 万点建索引 < 5s，单次查询 < 5ms，
索引常驻内存 < 200MB，查询不改变索引占用。

这些断言用的是 README 里的指标，在本机（普通开发机）留有数倍余量。
"""

import gc
import random
import sys
import time
import unittest

from spatial_index import KDTreeIndex

N = 500_000
BUILD_LIMIT_S = 5.0
QUERY_LIMIT_MS = 5.0
MEMORY_LIMIT_BYTES = 200 * 1024 * 1024


def _index_footprint(idx):
    total = 0
    for name in ("_lat", "_lon", "_pid", "_left", "_right", "_size",
                 "_alive", "_free"):
        total += sys.getsizeof(getattr(idx, name))
    index_map = idx._index
    total += sys.getsizeof(index_map)
    for key, value in index_map.items():
        total += sys.getsizeof(key) + sys.getsizeof(value)
    return total


class PerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(2024)
        cls.pts = [
            (i, rng.randint(18000000, 54000000), rng.randint(73000000, 135000000))
            for i in range(1, N + 1)
        ]
        cls.rng = random.Random(99)

    def test_1_build_under_5s(self):
        t0 = time.perf_counter()
        type(self).idx = KDTreeIndex(self.pts)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, BUILD_LIMIT_S,
                        "build 500k took %.2fs" % elapsed)

    def test_2_rect_query_fast(self):
        rng = self.rng
        idx = type(self).idx
        rounds = 1000
        t0 = time.perf_counter()
        for _ in range(rounds):
            la = rng.randint(18000000, 50000000)
            lo = rng.randint(73000000, 130000000)
            idx.rect(la, lo, la + 2000000, lo + 2000000)
        avg_ms = (time.perf_counter() - t0) / rounds * 1000
        self.assertLess(avg_ms, QUERY_LIMIT_MS, "rect avg %.3fms" % avg_ms)

    def test_3_nearest_query_fast(self):
        rng = self.rng
        idx = type(self).idx
        rounds = 1000
        t0 = time.perf_counter()
        for _ in range(rounds):
            idx.nearest(rng.randint(18000000, 54000000),
                        rng.randint(73000000, 135000000), 100)
        avg_ms = (time.perf_counter() - t0) / rounds * 1000
        self.assertLess(avg_ms, QUERY_LIMIT_MS, "nearest avg %.3fms" % avg_ms)

    def test_4_updates_stay_fast(self):
        rng = self.rng
        idx = type(self).idx
        t0 = time.perf_counter()
        for _ in range(100000):
            idx.update(rng.randint(1, N),
                       rng.randint(18000000, 54000000),
                       rng.randint(73000000, 135000000))
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 30.0, "100k updates took %.2fs" % elapsed)
        # 高频更新之后查询不能变慢
        t0 = time.perf_counter()
        for _ in range(500):
            idx.nearest(rng.randint(18000000, 54000000),
                        rng.randint(73000000, 135000000), 100)
        avg_ms = (time.perf_counter() - t0) / 500 * 1000
        self.assertLess(avg_ms, QUERY_LIMIT_MS,
                        "nearest after churn avg %.3fms" % avg_ms)

    def test_5_memory_under_200mb(self):
        gc.collect()
        footprint = _index_footprint(type(self).idx)
        self.assertLess(footprint, MEMORY_LIMIT_BYTES,
                        "index footprint %.1f MB" % (footprint / 1e6))

    def test_6_queries_do_not_grow_index(self):
        rng = self.rng
        idx = type(self).idx
        gc.collect()
        before = (len(idx._pid), len(idx._index), len(idx._free),
                  idx._live, idx._dead, _index_footprint(idx))
        for _ in range(50000):
            idx.nearest(rng.randint(18000000, 54000000),
                        rng.randint(73000000, 135000000), 10)
            idx.rect(30000000, 100000000, 31000000, 101000000)
        gc.collect()
        after = (len(idx._pid), len(idx._index), len(idx._free),
                 idx._live, idx._dead, _index_footprint(idx))
        self.assertEqual(before, after, "queries must not grow the index")


if __name__ == "__main__":
    unittest.main()
