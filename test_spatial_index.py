"""SpatialIndex 的 unittest 测试套件。

覆盖：样例对拍、边界口径、异常、随机对拍（硬要求）、
增删 churn 后无泄漏、查询不改变索引占用、性能量级。
"""

import os
import random
import time
import unittest

from brute_force import BruteForceIndex
from run_case import format_result, load_ops, load_points, run
from spatial_index import SpatialIndex

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


class SampleCaseTest(unittest.TestCase):
    """samples/ 下每组 case 与 expected 逐行一致。"""

    def test_all_cases(self):
        for name in sorted(os.listdir(SAMPLES)):
            if not name.endswith(".points.csv"):
                continue
            case = os.path.join(SAMPLES, name[: -len(".points.csv")])
            with self.subTest(case=os.path.basename(case)):
                results = run(SpatialIndex(), load_points(case + ".points.csv"),
                              load_ops(case + ".ops.csv"))
                got = [format_result(i + 1, ids) for i, ids in enumerate(results)]
                with open(case + ".expected.csv") as f:
                    want = [ln.strip() for ln in f if ln.strip()]
                self.assertEqual(got, want)


class BoundaryTest(unittest.TestCase):
    def setUp(self):
        self.idx = SpatialIndex()

    def test_rect_closed_interval(self):
        # 四条边和四个角上的点都算在内
        self.idx.insert(1, 0, 0)
        self.idx.insert(2, 0, 100)
        self.idx.insert(3, 100, 0)
        self.idx.insert(4, 100, 100)
        self.idx.insert(5, 50, 0)      # 边上
        self.idx.insert(6, 50, 50)     # 内部
        self.idx.insert(7, 101, 50)    # 外部
        self.assertEqual(self.idx.rect(0, 0, 100, 100), [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.idx.rect(0, 0, 0, 0), [1])          # 单点矩形
        self.assertEqual(self.idx.rect(100, 100, 100, 100), [4])

    def test_rect_reversed_is_empty(self):
        self.idx.insert(1, 0, 0)
        self.assertEqual(self.idx.rect(10, 0, 0, 10), [])
        self.assertEqual(self.idx.rect(0, 10, 10, 0), [])
        self.assertEqual(self.idx.rect(10, 10, 0, 0), [])

    def test_rect_result_sorted_by_id(self):
        for bid in (9, 3, 7, 1):
            self.idx.insert(bid, bid, bid)
        self.assertEqual(self.idx.rect(-10, -10, 100, 100), [1, 3, 7, 9])

    def test_nearest_tie_breaks_by_id(self):
        self.idx.insert(5, 0, 100)
        self.idx.insert(2, 0, -100)
        self.idx.insert(9, 100, 0)
        # 三辆距离平方相同，按编号升序
        self.assertEqual(self.idx.nearest(0, 0, 3), [2, 5, 9])

    def test_nearest_same_location(self):
        # 同一位置多辆车各自独立返回，按编号定序
        for bid in (4, 1, 3):
            self.idx.insert(bid, 500, 500)
        self.assertEqual(self.idx.nearest(500, 500, 2), [1, 3])
        self.assertEqual(self.idx.nearest(500, 500, 10), [1, 3, 4])

    def test_nearest_k_edge(self):
        self.idx.insert(1, 0, 0)
        self.assertEqual(self.idx.nearest(0, 0, 0), [])
        self.assertEqual(self.idx.nearest(0, 0, -3), [])
        self.assertEqual(self.idx.nearest(0, 0, 100), [1])  # 不足 k 返回全部
        self.assertEqual(SpatialIndex().nearest(0, 0, 5), [])

    def test_negative_coords(self):
        self.idx.insert(1, -90_000_000, -180_000_000)
        self.idx.insert(2, 90_000_000, 180_000_000)
        self.assertEqual(self.idx.rect(-90_000_000, -180_000_000, 0, 0), [1])
        self.assertEqual(self.idx.nearest(-90_000_000, -180_000_000, 1), [1])

    def test_errors(self):
        self.idx.insert(1, 0, 0)
        with self.assertRaises(ValueError):
            self.idx.insert(1, 5, 5)          # 重复编号
        with self.assertRaises(ValueError):
            self.idx.remove(99)               # 不存在
        with self.assertRaises(ValueError):
            self.idx.update(99, 0, 0)         # 不存在
        with self.assertRaises(ValueError):
            self.idx.insert(2, 90_000_001, 0)  # 纬度越界
        with self.assertRaises(ValueError):
            self.idx.insert(3, 0, -180_000_001)  # 经度越界
        with self.assertRaises(ValueError):
            self.idx.insert(-1, 0, 0)         # 编号非正
        # 失败的 update 不能把原来的车弄丢
        with self.assertRaises(ValueError):
            self.idx.update(1, 0, 180_000_001)
        self.assertEqual(self.idx.nearest(0, 0, 1), [1])

    def test_update_moves_bike(self):
        self.idx.insert(1, 0, 0)
        self.idx.update(1, 1000, 1000)
        self.assertEqual(self.idx.rect(-1, -1, 1, 1), [])
        self.assertEqual(self.idx.rect(999, 999, 1001, 1001), [1])
        self.assertEqual(len(self.idx), 1)

    def test_remove_then_reinsert(self):
        self.idx.insert(1, 0, 0)
        self.idx.remove(1)
        self.assertEqual(self.idx.rect(-1, -1, 1, 1), [])
        self.idx.insert(1, 5, 5)
        self.assertEqual(self.idx.nearest(0, 0, 1), [1])


class RandomCrossCheckTest(unittest.TestCase):
    """硬要求：任意增删改序列下，两种查询与暴力扫描逐项一致。"""

    def run_cross_check(self, seed, rounds, coord_pool):
        rng = random.Random(seed)
        idx, ref = SpatialIndex(), BruteForceIndex()
        live = set()
        next_bid = 1
        for step in range(rounds):
            op = rng.random()
            if op < 0.35 or not live:
                bid = next_bid
                next_bid += 1
                lat, lon = coord_pool(rng)
                idx.insert(bid, lat, lon)
                ref.insert(bid, lat, lon)
                live.add(bid)
            elif op < 0.55:
                bid = rng.choice(tuple(live))
                idx.remove(bid)
                ref.remove(bid)
                live.discard(bid)
            elif op < 0.70:
                bid = rng.choice(tuple(live))
                lat, lon = coord_pool(rng)
                idx.update(bid, lat, lon)
                ref.update(bid, lat, lon)
            elif op < 0.85:
                a, b = coord_pool(rng), coord_pool(rng)
                min_lat, max_lat = sorted((a[0], b[0]))
                min_lon, max_lon = sorted((a[1], b[1]))
                if rng.random() < 0.1:  # 偶尔反向矩形
                    min_lat, max_lat = max_lat + 1, min_lat
                self.assertEqual(idx.rect(min_lat, min_lon, max_lat, max_lon),
                                 ref.rect(min_lat, min_lon, max_lat, max_lon),
                                 "rect 不一致 @step %d" % step)
            else:
                lat, lon = coord_pool(rng)
                k = rng.randint(0, 20)
                self.assertEqual(idx.nearest(lat, lon, k),
                                 ref.nearest(lat, lon, k),
                                 "nearest 不一致 @step %d" % step)
        return idx

    def test_random_uniform(self):
        def pool(rng):
            return (rng.randint(-90_000_000, 90_000_000),
                    rng.randint(-180_000_000, 180_000_000))
        self.run_cross_check(seed=20260924, rounds=4000, coord_pool=pool)

    def test_random_clustered_with_duplicates(self):
        # 小范围密集坐标，大量同位置车辆，专测并列与重复坐标
        def pool(rng):
            return (rng.randint(-500, 500), rng.randint(-500, 500))
        self.run_cross_check(seed=7, rounds=6000, coord_pool=pool)

    def test_random_china_scale(self):
        def pool(rng):
            return (rng.randint(18_000_000, 54_000_000),
                    rng.randint(73_000_000, 136_000_000))
        self.run_cross_check(seed=123456, rounds=4000, coord_pool=pool)


class ChurnStabilityTest(unittest.TestCase):
    """频繁增删不退化、无泄漏：结构节点数始终与存活数同量级。"""

    def test_churn_keeps_structure_bounded(self):
        rng = random.Random(99)
        idx, ref = SpatialIndex(), BruteForceIndex()
        live = []
        for step in range(60000):
            if not live or rng.random() < 0.5:
                bid = step + 1
                lat, lon = rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6)
                idx.insert(bid, lat, lon)
                ref.insert(bid, lat, lon)
                live.append(bid)
            else:
                bid = live.pop(rng.randrange(len(live)))
                if rng.random() < 0.5:
                    idx.remove(bid)
                    ref.remove(bid)
                else:
                    lat, lon = rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6)
                    idx.update(bid, lat, lon)
                    ref.update(bid, lat, lon)
                    live.append(bid)
        n_live, n_dead, n_total = idx.stats()
        self.assertEqual(n_live, len(ref))
        # 墓碑被定期重建回收，永远不超过存活数，总量有界
        self.assertLessEqual(n_dead, max(1, n_live))
        self.assertLessEqual(n_total, 2 * max(1, n_live))
        # churn 之后查询仍然正确
        for _ in range(200):
            lat, lon = rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6)
            self.assertEqual(idx.nearest(lat, lon, 10), ref.nearest(lat, lon, 10))
            self.assertEqual(idx.rect(lat - 5000, lon - 5000, lat + 5000, lon + 5000),
                             ref.rect(lat - 5000, lon - 5000, lat + 5000, lon + 5000))

    def test_queries_do_not_grow_index(self):
        rng = random.Random(42)
        idx = SpatialIndex()
        for bid in range(1, 20001):
            idx.insert(bid, rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6))
        before = idx.stats()
        for _ in range(200000):
            lat, lon = rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6)
            idx.nearest(lat, lon, 10)
            idx.rect(lat - 3000, lon - 3000, lat + 3000, lon + 3000)
        self.assertEqual(idx.stats(), before)  # 查询不改变索引占用


class PerformanceTest(unittest.TestCase):
    """量级验证：50 万点建索引 < 5s，单次查询 < 5ms。"""

    N = 500_000

    @classmethod
    def setUpClass(cls):
        rng = random.Random(2026)
        cls.points = [
            (bid, rng.randint(18_000_000, 54_000_000),
             rng.randint(73_000_000, 136_000_000))
            for bid in range(1, cls.N + 1)
        ]
        t0 = time.perf_counter()
        cls.idx = SpatialIndex()
        for bid, lat, lon in cls.points:
            cls.idx.insert(bid, lat, lon)
        cls.build_seconds = time.perf_counter() - t0

    def test_build_under_5s(self):
        self.assertLess(self.build_seconds, 5.0,
                        "建索引耗时 %.2fs" % self.build_seconds)

    def test_nearest_under_5ms(self):
        rng = random.Random(1)
        queries = [(rng.randint(18_000_000, 54_000_000),
                    rng.randint(73_000_000, 136_000_000), 100)
                   for _ in range(500)]
        t0 = time.perf_counter()
        for lat, lon, k in queries:
            self.idx.nearest(lat, lon, k)
        avg_ms = (time.perf_counter() - t0) / len(queries) * 1000
        self.assertLess(avg_ms, 5.0, "nearest 平均 %.3fms" % avg_ms)

    def test_rect_under_5ms(self):
        rng = random.Random(2)
        queries = []
        for _ in range(500):
            lat = rng.randint(18_000_000, 54_000_000)
            lon = rng.randint(73_000_000, 136_000_000)
            queries.append((lat - 50_000, lon - 50_000,
                            lat + 50_000, lon + 50_000))
        t0 = time.perf_counter()
        for q in queries:
            self.idx.rect(*q)
        avg_ms = (time.perf_counter() - t0) / len(queries) * 1000
        self.assertLess(avg_ms, 5.0, "rect 平均 %.3fms" % avg_ms)


if __name__ == "__main__":
    unittest.main(verbosity=2)
