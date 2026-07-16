import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1] / "app"
ZHUZHU_DIR = APP_DIR / "zhuzhu"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ZHUZHU_DIR))

import generate_gptimage  # noqa: E402


class GptImageFailureTest(unittest.TestCase):
    def tearDown(self):
        generate_gptimage._LAST_TERMINAL_IMAGE_FAILURE = ""
        generate_gptimage._IMAGES_API_UNSUPPORTED_BASES.clear()

    def test_quota_failure_is_terminal_and_is_not_retried(self):
        response = SimpleNamespace(
            status_code=429,
            text='{"error":{"code":"insufficient_quota","message":"图片账号额度已用完"}}',
        )

        with patch.object(
            generate_gptimage.REQUEST_SESSION,
            "post",
            return_value=response,
        ) as post, patch.object(generate_gptimage.time, "sleep") as sleep:
            result = generate_gptimage._generate_via_images_api(
                "a red apple",
                None,
                "1024x1024",
                "http://example.test/v1",
            )

        self.assertIsNone(result)
        self.assertEqual(1, post.call_count)
        sleep.assert_not_called()
        self.assertEqual(
            "GPT Image 图片账号额度已用完",
            generate_gptimage._LAST_TERMINAL_IMAGE_FAILURE,
        )

    def test_terminal_img2img_failure_skips_text2img_retry(self):
        def fail_once(*_args, **_kwargs):
            generate_gptimage._LAST_TERMINAL_IMAGE_FAILURE = "GPT Image 没有可用渠道"
            return None

        with patch.object(
            generate_gptimage,
            "_generate_via_direct_gpt",
            side_effect=fail_once,
        ) as direct:
            result = generate_gptimage.generate(
                "custom",
                prompt_override="adult portrait",
                ref_image="/tmp/reference.png",
                prompt_is_final=True,
                sync_gallery=False,
            )

        self.assertIsNone(result)
        self.assertEqual(1, direct.call_count)

    def test_precision_edit_never_retries_without_reference_image(self):
        with patch.object(
            generate_gptimage,
            "_generate_via_direct_gpt",
            return_value=None,
        ) as direct:
            result = generate_gptimage.generate(
                "custom",
                prompt_override="change only the background",
                ref_image="/tmp/reference.png",
                prompt_is_final=True,
                sync_gallery=False,
                precise_edit=True,
            )

        self.assertIsNone(result)
        self.assertEqual(1, direct.call_count)
        self.assertEqual("/tmp/reference.png", direct.call_args.args[1])
        self.assertTrue(direct.call_args.kwargs["precise_edit"])

    def test_terminal_failure_reason_covers_current_axonhub_errors(self):
        cases = {
            '{"code":"insufficient_quota"}': "GPT Image 图片账号额度已用完",
            '{"detail":{"code":"deactivated_workspace"}}': "GPT Image 工作区已停用",
            '{"message":"No available compatible accounts"}': "GPT Image 没有可用的兼容账号",
            '{"code":"model_not_found","message":"No available channel"}': "GPT Image 没有可用渠道",
        }

        for body, expected in cases.items():
            with self.subTest(body=body):
                self.assertEqual(
                    expected,
                    generate_gptimage._terminal_image_failure_reason(429, body),
                )

    def test_images_failure_never_falls_back_to_chat(self):
        with patch.object(
            generate_gptimage,
            "_get_gpt_raw_base_url",
            return_value="http://example.test/v1",
        ), patch.object(
            generate_gptimage,
            "GPTIMAGE_DIRECT_MODEL",
            "gpt-image-2",
        ), patch.object(
            generate_gptimage,
            "_generate_via_images_api",
            return_value=None,
        ) as images_api, patch.object(generate_gptimage.REQUEST_SESSION, "post") as post:
            result = generate_gptimage._generate_via_direct_gpt("portrait")

        self.assertIsNone(result)
        images_api.assert_called_once_with(
            "portrait",
            None,
            None,
            "http://example.test/v1",
            precise_edit=False,
        )
        post.assert_not_called()

    def test_chat_fallback_hooks_are_removed(self):
        self.assertFalse(hasattr(generate_gptimage, "_gpt_chat_fallback_enabled"))
        self.assertFalse(hasattr(generate_gptimage, "_generate_via_chat_gpt"))

    def test_explicit_chat_endpoint_is_rewritten_to_images(self):
        response = SimpleNamespace(status_code=400, text="bad request")
        with patch.object(
            generate_gptimage,
            "_get_gpt_raw_base_url",
            return_value="http://example.test/v1/chat/completions",
        ), patch.object(
            generate_gptimage,
            "GPTIMAGE_DIRECT_MODEL",
            "gpt-image-2",
        ), patch.object(
            generate_gptimage.REQUEST_SESSION,
            "post",
            return_value=response,
        ) as post:
            result = generate_gptimage._generate_via_direct_gpt("portrait")

        self.assertIsNone(result)
        post.assert_called_once()
        self.assertEqual(
            "http://example.test/v1/images/generations",
            post.call_args.args[0],
        )
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)


if __name__ == "__main__":
    unittest.main()
