"""样例运行器：python3 run_case.py samples/case-1

按 points -> ops 的顺序执行，把每个查询的结果打到标准输出，
并和 expected 文件逐行对比（有 expected 时）。
"""

import sys

from spatial_index import SpatialIndex


def load_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bid, lat, lon = line.split(",")
            pts.append((int(bid), int(lat), int(lon)))
    return pts


def load_ops(path):
    ops = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            ops.append((parts[0], [int(x) for x in parts[1:]]))
    return ops


def run(index, points, ops):
    """执行操作序列，返回每个查询的结果列表（编号列表）。"""
    for bid, lat, lon in points:
        index.insert(bid, lat, lon)
    results = []
    for name, args in ops:
        if name == "insert":
            index.insert(*args)
        elif name == "remove":
            index.remove(*args)
        elif name == "update":
            index.update(*args)
        elif name == "rect":
            results.append(index.rect(*args))
        elif name == "nearest":
            results.append(index.nearest(*args))
        else:
            raise ValueError("未知操作: %s" % name)
    return results


def format_result(seq, ids):
    return "%d,%s" % (seq, ",".join(map(str, ids)) if ids else "-")


def main(argv):
    case = argv[1]
    points = load_points(case + ".points.csv")
    ops = load_ops(case + ".ops.csv")
    results = run(SpatialIndex(), points, ops)
    lines = [format_result(i + 1, ids) for i, ids in enumerate(results)]
    try:
        with open(case + ".expected.csv") as f:
            expected = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        expected = None
    for line in lines:
        print(line)
    if expected is not None:
        if lines == expected:
            print("OK: 与 %s.expected.csv 完全一致" % case, file=sys.stderr)
            return 0
        print("FAIL: 与期望不一致", file=sys.stderr)
        for got, want in zip(lines, expected):
            if got != want:
                print("  got:  %s\n  want: %s" % (got, want), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
