"""样例文件驱动：读取 points/ops 文件，执行操作，返回每个查询的结果。

文件格式见 README：points 是 `id,lat_e6,lon_e6`，ops 每行一个操作，
`#` 开头是注释。每个 rect/nearest 查询产生一条结果（编号列表）。
"""

from .kdtree import KDTreeIndex


def _rows(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                yield [field.strip() for field in line.split(",")]


def load_points(path):
    return [(int(row[0]), int(row[1]), int(row[2])) for row in _rows(path)]


def run_ops(points_path, ops_path):
    """按 ops 文件顺序执行，返回每个查询的编号列表（按查询顺序）。"""
    index = KDTreeIndex(load_points(points_path))
    results = []
    for row in _rows(ops_path):
        op = row[0]
        if op == "insert":
            index.insert(int(row[1]), int(row[2]), int(row[3]))
        elif op == "remove":
            index.remove(int(row[1]))
        elif op == "update":
            index.update(int(row[1]), int(row[2]), int(row[3]))
        elif op == "rect":
            results.append(
                index.rect(int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            )
        elif op == "nearest":
            results.append(
                index.nearest(int(row[1]), int(row[2]), int(row[3]))
            )
        else:
            raise ValueError("unknown op: %r" % (op,))
    return results


def format_results(results):
    """格式化成 expected 文件的样子：`序号,id1,id2,...`，空结果写 `-`。"""
    lines = []
    for seq, ids in enumerate(results, 1):
        lines.append(",".join([str(seq)] + [str(i) for i in ids]) if ids else "%d,-" % seq)
    return "\n".join(lines) + "\n"
