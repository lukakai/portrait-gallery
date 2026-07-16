import asyncio
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from data import DailyEntry  # noqa: E402
from main import PortraitGalleryApp  # noqa: E402
from web_server import GalleryServer  # noqa: E402


class DummyRequest:
    can_read_body = False


class ScheduleRecoveryTest(unittest.IsolatedAsyncioTestCase):
    def test_generate_now_passes_the_same_persisted_detail_to_the_cli(self):
        server = GalleryServer.__new__(GalleryServer)
        now = datetime(2026, 7, 16, 1, 0)
        daily = {
            "schedule_details": [
                {
                    "time": "23:00",
                    "activity_zh": "在窗边阅读睡前小说",
                    "activity_en": "reading a bedtime novel by the window",
                    "scene_en": "quiet bedroom window seat",
                }
            ]
        }

        detail = server._schedule_generate_now_detail(now, daily)
        activity = f"{detail['activity_zh']}，盖好薄毯"
        detail["activity_zh"] = activity
        schedule_time, args = server._generate_now_schedule_args(
            "01:00",
            activity,
            detail,
        )

        self.assertEqual("01:00 在窗边阅读睡前小说，盖好薄毯", schedule_time)
        self.assertEqual("--schedule-time", args[0])
        self.assertEqual(schedule_time, args[1])
        self.assertEqual("--schedule-detail-json", args[2])
        self.assertEqual(detail, json.loads(args[3]))

    async def test_app_schedule_refresh_is_singleflight_for_concurrent_callers(self):
        app = PortraitGalleryApp.__new__(PortraitGalleryApp)
        app._schedule_refresh_task = None
        started = asyncio.Event()
        release = asyncio.Event()
        entry = DailyEntry(date="2026-07-15", schedule="08:00 买早餐", status="ok")
        calls = 0

        async def refresh_impl():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return entry

        app._refresh_schedule_impl = refresh_impl
        first = asyncio.create_task(app.refresh_schedule())
        await started.wait()
        second = asyncio.create_task(app.refresh_schedule())
        await asyncio.sleep(0)
        release.set()

        first_result, second_result = await asyncio.gather(first, second)

        self.assertIs(entry, first_result)
        self.assertIs(entry, second_result)
        self.assertEqual(1, calls)
        self.assertIsNone(app._schedule_refresh_task)

    async def test_startup_missing_schedule_always_starts_background_generation(self):
        app = PortraitGalleryApp.__new__(PortraitGalleryApp)
        app._today_schedule_entry = Mock(return_value=None)
        app.daily_job = AsyncMock()

        await app._restore_daily_schedule_state()
        await asyncio.sleep(0)

        app.daily_job.assert_awaited_once()

    async def test_generate_now_refreshes_missing_schedule_before_returning(self):
        server = GalleryServer.__new__(GalleryServer)
        refreshed = DailyEntry(
            date="2026-07-15",
            schedule="08:00 买早餐",
            status="ok",
        )
        server.on_refresh_schedule = AsyncMock(return_value=refreshed)
        server._schedule_refresh_task = None
        server._today_schedule_entry = Mock(side_effect=[{}, refreshed.to_dict()])
        server._schedule_items_for_inference = Mock(return_value=[])

        response = await server.handle_generate_now(DummyRequest())
        payload = json.loads(response.text)

        server.on_refresh_schedule.assert_awaited_once()
        self.assertEqual(400, response.status)
        self.assertEqual("schedule_time_not_found", payload.get("error"))


if __name__ == "__main__":
    unittest.main()
