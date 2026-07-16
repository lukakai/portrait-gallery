import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from settings import (  # noqa: E402
    sanitize_schedule_forbidden_text,
    schedule_forbidden_negative_clause,
    schedule_forbidden_variants,
)


class ScheduleForbiddenAccessoriesTest(unittest.TestCase):
    def test_cross_star_necklace_variants_are_blocked(self):
        variants = schedule_forbidden_variants("银色十字星锁骨链")

        self.assertIn("十字星项链", variants)
        self.assertIn("cross-star necklace", variants)
        self.assertIn("silver collarbone chain", variants)
        self.assertEqual(
            "穿白色衬衫",
            sanitize_schedule_forbidden_text(
                "穿白色衬衫，佩戴十字星项链",
                ["银色十字星锁骨链"],
            ),
        )

    def test_bag_and_cross_star_visual_exclusions_are_both_kept(self):
        clause = schedule_forbidden_negative_clause(["包", "银色十字星锁骨链"])

        self.assertIn("no bags", clause)
        self.assertIn("no silver collarbone chain", clause)
        self.assertIn("no silver cross-star necklace", clause)


if __name__ == "__main__":
    unittest.main()
