"""随机对拍：任意增删改序列之后，查询结果必须和暴力全扫逐项一致。"""

import random
import unittest

from spatial_index import KDTreeIndex


class BruteForce:
    """暴力参照实现：字典存点，查询全扫。"""

    def __init__(self):
        self.pts = {}

    def insert(self, pid, lat, lon):
        assert pid not in self.pts
        self.pts[pid] = (lat, lon)

    def remove(self, pid):
        del self.pts[pid]

    def update(self, pid, lat, lon):
        assert pid in self.pts
        self.pts[pid] = (lat, lon)

    def rect(self, min_lat, min_lon, max_lat, max_lon):
        if min_lat > max_lat or min_lon > max_lon:
            return []
        return sorted(
            pid
            for pid, (lat, lon) in self.pts.items()
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
        )

    def nearest(self, lat, lon, k):
        if k <= 0:
            return []
        keyed = sorted(
            ((la - lat) ** 2 + (lo - lon) ** 2, pid)
            for pid, (la, lo) in self.pts.items()
        )
        return [pid for _, pid in keyed[:k]]


class RandomDiffTest(unittest.TestCase):
    def run_round(self, seed, rounds, coord_pool):
        rng = random.Random(seed)
        index = KDTreeIndex()
        brute = BruteForce()
        live = set()
        next_id = 1
        for step in range(rounds):
            action = rng.random()
            if action < 0.30 or not live:
                pid = next_id
                next_id += 1
                lat, lon = rng.choice(coord_pool), rng.choice(coord_pool)
                index.insert(pid, lat, lon)
                brute.insert(pid, lat, lon)
                live.add(pid)
            elif action < 0.50:
                pid = rng.choice(tuple(live))
                index.remove(pid)
                brute.remove(pid)
                live.discard(pid)
            elif action < 0.70:
                pid = rng.choice(tuple(live))
                lat, lon = rng.choice(coord_pool), rng.choice(coord_pool)
                index.update(pid, lat, lon)
                brute.update(pid, lat, lon)
            elif action < 0.85:
                a, b = rng.choice(coord_pool), rng.choice(coord_pool)
                c, d = rng.choice(coord_pool), rng.choice(coord_pool)
                args = (min(a, b), min(c, d), max(a, b), max(c, d))
                if rng.random() < 0.1:  # 偶尔故意反向
                    args = (args[2], args[3], args[0], args[1])
                self.assertEqual(index.rect(*args), brute.rect(*args),
                                 "rect%r seed=%d step=%d" % (args, seed, step))
            else:
                lat, lon = rng.choice(coord_pool), rng.choice(coord_pool)
                k = rng.randint(0, 12)
                self.assertEqual(index.nearest(lat, lon, k),
                                 brute.nearest(lat, lon, k),
                                 "nearest(%r,%r,%r) seed=%d step=%d"
                                 % (lat, lon, k, seed, step))
        # 收尾再全量对拍几次
        for _ in range(20):
            lat, lon = rng.choice(coord_pool), rng.choice(coord_pool)
            k = rng.randint(1, 50)
            self.assertEqual(index.nearest(lat, lon, k), brute.nearest(lat, lon, k))
            a, b = rng.choice(coord_pool), rng.choice(coord_pool)
            c, d = rng.choice(coord_pool), rng.choice(coord_pool)
            args = (min(a, b), min(c, d), max(a, b), max(c, d))
            self.assertEqual(index.rect(*args), brute.rect(*args))
        self.assertEqual(len(index), len(brute.pts))

    def test_dense_coords(self):
        # 坐标池小，大量同坐标点，专考并列距离与同位多车
        pool = list(range(-100, 101, 10))
        for seed in range(5):
            with self.subTest(seed=seed):
                self.run_round(seed, 800, pool)

    def test_sparse_coords(self):
        rng = random.Random(999)
        pool = [rng.randint(-90000000, 90000000) for _ in range(2000)]
        for seed in range(5, 9):
            with self.subTest(seed=seed):
                self.run_round(seed, 800, pool)

    def test_china_range(self):
        rng = random.Random(7)
        pool = [rng.randint(18000000, 54000000) for _ in range(500)]
        for seed in range(9, 12):
            with self.subTest(seed=seed):
                self.run_round(seed, 600, pool)


if __name__ == "__main__":
    unittest.main()
