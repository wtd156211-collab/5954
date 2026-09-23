import os
import unittest

from spatial_index.runner import run_ops, format_results

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


class SampleCasesTest(unittest.TestCase):
    def test_all_cases(self):
        for case in (1, 2, 3, 4):
            with self.subTest(case=case):
                results = run_ops(
                    os.path.join(SAMPLES, "case-%d.points.csv" % case),
                    os.path.join(SAMPLES, "case-%d.ops.csv" % case),
                )
                with open(
                    os.path.join(SAMPLES, "case-%d.expected.csv" % case),
                    encoding="utf-8",
                ) as fh:
                    expected = fh.read()
                self.assertEqual(format_results(results), expected)


if __name__ == "__main__":
    unittest.main()
