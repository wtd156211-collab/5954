import unittest

from spatial_index import KDTreeIndex


class EdgeCasesTest(unittest.TestCase):
    def test_empty_index(self):
        idx = KDTreeIndex()
        self.assertEqual(idx.rect(-90000000, -180000000, 90000000, 180000000), [])
        self.assertEqual(idx.nearest(0, 0, 10), [])
        self.assertEqual(len(idx), 0)

    def test_reversed_rect(self):
        idx = KDTreeIndex([(1, 0, 0)])
        self.assertEqual(idx.rect(5, 0, -5, 10), [])
        self.assertEqual(idx.rect(0, 5, 10, -5), [])

    def test_boundary_inclusive(self):
        idx = KDTreeIndex([(1, 0, 0), (2, 10, 10), (3, 0, 10), (4, 10, 0), (5, 5, 5)])
        # 四条边和四个角都压线，闭区间全算
        self.assertEqual(idx.rect(0, 0, 10, 10), [1, 2, 3, 4, 5])
        self.assertEqual(idx.rect(0, 0, 0, 0), [1])
        self.assertEqual(idx.rect(10, 10, 10, 10), [2])

    def test_k_nonpositive(self):
        idx = KDTreeIndex([(1, 0, 0)])
        self.assertEqual(idx.nearest(0, 0, 0), [])
        self.assertEqual(idx.nearest(0, 0, -3), [])

    def test_k_larger_than_size(self):
        idx = KDTreeIndex([(1, 0, 0), (2, 3, 4)])
        self.assertEqual(idx.nearest(0, 0, 100), [1, 2])

    def test_tie_break_by_id(self):
        # 同距离并列按编号升序；同坐标多车各自独立返回
        idx = KDTreeIndex([(9, 0, 0), (3, 0, 0), (7, 10, 0), (1, -10, 0)])
        self.assertEqual(idx.nearest(0, 0, 4), [3, 9, 1, 7])
        self.assertEqual(idx.nearest(0, 0, 2), [3, 9])

    def test_duplicate_insert_raises(self):
        idx = KDTreeIndex([(1, 0, 0)])
        with self.assertRaises(KeyError):
            idx.insert(1, 5, 5)

    def test_remove_missing_raises(self):
        idx = KDTreeIndex([(1, 0, 0)])
        with self.assertRaises(KeyError):
            idx.remove(2)

    def test_update_missing_raises(self):
        idx = KDTreeIndex([(1, 0, 0)])
        with self.assertRaises(KeyError):
            idx.update(2, 5, 5)

    def test_remove_then_reinsert(self):
        idx = KDTreeIndex([(1, 0, 0), (2, 1, 1)])
        idx.remove(1)
        idx.insert(1, 9, 9)
        self.assertEqual(idx.rect(0, 0, 10, 10), [1, 2])
        self.assertEqual(idx.nearest(9, 9, 1), [1])

    def test_update_moves_point(self):
        idx = KDTreeIndex([(1, 0, 0), (2, 100, 100)])
        idx.update(1, 200, 200)
        self.assertEqual(idx.rect(0, 0, 50, 50), [])
        self.assertEqual(idx.rect(150, 150, 250, 250), [1])

    def test_coord_range_check(self):
        idx = KDTreeIndex()
        with self.assertRaises(ValueError):
            idx.insert(1, 90000001, 0)
        with self.assertRaises(ValueError):
            idx.insert(1, 0, -180000001)
        idx.insert(1, -90000000, 180000000)  # 边界值合法

    def test_negative_coords(self):
        idx = KDTreeIndex([(1, -5000000, -120000000), (2, 5000000, 120000000)])
        self.assertEqual(idx.rect(-90000000, -180000000, 0, 0), [1])
        self.assertEqual(idx.nearest(-5000000, -120000000, 1), [1])

    def test_heavy_churn_no_bloat(self):
        # 频繁增删改之后，内部槽位数应与存活点数同量级
        import random
        rng = random.Random(42)
        idx = KDTreeIndex((i, rng.randint(-1000, 1000), rng.randint(-1000, 1000))
                          for i in range(1, 2001))
        live = set(range(1, 2001))
        next_id = 2001
        for _ in range(20000):
            pid = rng.choice(tuple(live))
            idx.remove(pid)
            live.discard(pid)
            idx.insert(next_id, rng.randint(-1000, 1000), rng.randint(-1000, 1000))
            live.add(next_id)
            next_id += 1
        self.assertEqual(len(idx), 2000)
        # 槽位总数（含空闲链）不允许无限膨胀
        self.assertLessEqual(len(idx._pid), 8000)
        self.assertEqual(len(idx._index), 2000)


if __name__ == "__main__":
    unittest.main()
