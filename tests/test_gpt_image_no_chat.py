import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ZHUZHU_DIR = Path(__file__).resolve().parents[1] / "app" / "zhuzhu"
sys.path.insert(0, str(ZHUZHU_DIR))

import generate_gptimage  # noqa: E402


class GptImageNoChatTest(unittest.TestCase):
    def test_images_failure_never_falls_back_to_chat_for_any_image_model(self):
        for model in ("gpt-image-1", "gpt-image-2", "agnes-image-1", "custom-image-model"):
            with self.subTest(model=model):
                with (
                    patch.object(generate_gptimage, "GPTIMAGE_DIRECT_MODEL", model),
                    patch.object(
                        generate_gptimage,
                        "_get_gpt_raw_base_url",
                        return_value="https://example.test/v1",
                    ),
                    patch.object(
                        generate_gptimage,
                        "_generate_via_images_api",
                        return_value=None,
                    ) as images_api,
                    patch.object(generate_gptimage.REQUEST_SESSION, "post") as post,
                ):
                    result = generate_gptimage._generate_via_direct_gpt("draw a portrait")

                self.assertIsNone(result)
                images_api.assert_called_once_with(
                    "draw a portrait",
                    None,
                    None,
                    "https://example.test/v1",
                    precise_edit=False,
                    ref_images=None,
                )
                post.assert_not_called()

    def test_chat_url_configuration_is_rewritten_to_images_api(self):
        response = Mock(status_code=400, text="bad request")
        with (
            patch.object(generate_gptimage, "GPTIMAGE_DIRECT_MODEL", "gpt-image-2"),
            patch.object(
                generate_gptimage,
                "_get_gpt_raw_base_url",
                return_value="https://example.test/v1/chat/completions",
            ),
            patch.object(generate_gptimage.REQUEST_SESSION, "post", return_value=response) as post,
        ):
            result = generate_gptimage._generate_via_direct_gpt("draw a portrait")

        self.assertIsNone(result)
        post.assert_called_once()
        self.assertEqual(
            "https://example.test/v1/images/generations",
            post.call_args.args[0],
        )
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)

    def test_chat_url_with_query_and_fragment_is_rewritten_to_images_api(self):
        self.assertEqual(
            "https://example.test/v1/images/generations?token=abc",
            generate_gptimage._gpt_images_endpoint(
                "https://example.test/v1/chat/completions?token=abc#ignored",
                "images/generations",
            ),
        )

    def test_non_images_route_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_gptimage._gpt_images_endpoint(
                "https://example.test/v1",
                "chat/completions",
            )

    def test_http_redirect_to_chat_is_not_followed(self):
        response = Mock(
            status_code=307,
            text="temporary redirect",
            headers={"Location": "https://example.test/v1/chat/completions"},
        )
        with (
            patch.object(generate_gptimage, "GPTIMAGE_DIRECT_MODEL", "gpt-image-2"),
            patch.object(
                generate_gptimage,
                "_get_gpt_raw_base_url",
                return_value="https://example.test/v1",
            ),
            patch.object(generate_gptimage.REQUEST_SESSION, "post", return_value=response) as post,
        ):
            result = generate_gptimage._generate_via_direct_gpt("draw a portrait")

        self.assertIsNone(result)
        post.assert_called_once()
        self.assertEqual(
            "https://example.test/v1/images/generations",
            post.call_args.args[0],
        )
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)

    def test_image_download_redirect_to_chat_is_not_followed(self):
        response = Mock(
            status_code=302,
            headers={"Location": "https://example.test/v1/chat/completions"},
        )
        with patch.object(
            generate_gptimage.REQUEST_SESSION,
            "get",
            return_value=response,
        ) as get:
            with self.assertRaisesRegex(RuntimeError, "Chat endpoint is disabled"):
                generate_gptimage._image_response_bytes(
                    {"data": [{"url": "https://cdn.example.test/image.png"}]}
                )

        get.assert_called_once_with(
            "https://cdn.example.test/image.png",
            timeout=60,
            allow_redirects=False,
        )

    def test_img2img_may_fall_back_only_to_images_text2img(self):
        with (
            patch.object(
                generate_gptimage,
                "_generate_via_direct_gpt",
                side_effect=[None, (b"image", 0.1)],
            ) as direct,
            patch.object(generate_gptimage, "_gpt_endpoint_label", return_value="example.test"),
            patch.object(
                generate_gptimage,
                "save_image",
                return_value=("/tmp/generated.png", "generated.png", "timestamp"),
            ),
            patch.object(generate_gptimage, "update_metadata"),
        ):
            result = generate_gptimage.generate(
                "custom",
                prompt_override="draw a portrait",
                ref_image="/tmp/reference.png",
                prompt_is_final=True,
                sync_gallery=False,
            )

        self.assertEqual("/tmp/generated.png", result)
        self.assertEqual(2, direct.call_count)
        self.assertEqual("/tmp/reference.png", direct.call_args_list[0].args[1])
        self.assertIsNone(direct.call_args_list[1].args[1])

    def test_reference_image_uses_edits_even_when_chat_url_is_configured(self):
        response = Mock(status_code=400, text="bad request")
        with (
            patch.object(generate_gptimage, "GPTIMAGE_DIRECT_MODEL", "gpt-image-2"),
            patch.object(
                generate_gptimage,
                "_get_gpt_raw_base_url",
                return_value="https://example.test/v1/chat/completions",
            ),
            patch.object(generate_gptimage, "_image_bytes_for_edit", return_value=b"image"),
            patch.object(generate_gptimage.REQUEST_SESSION, "post", return_value=response) as post,
        ):
            result = generate_gptimage._generate_via_direct_gpt(
                "edit this portrait", "/tmp/reference.png"
            )

        self.assertIsNone(result)
        post.assert_called_once()
        self.assertEqual(
            "https://example.test/v1/images/edits",
            post.call_args.args[0],
        )
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)


if __name__ == "__main__":
    unittest.main()
