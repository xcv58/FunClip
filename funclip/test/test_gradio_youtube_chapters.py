import asyncio
from collections import OrderedDict
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr


with patch("funasr.AutoModel", return_value=object()):
    import gradio_app


SIMPLE_SRT = "1\n00:00:00,000 --> 00:00:35,000\n简体中文字幕\n"


def staged_outputs(candidate):
    return tuple(candidate["outputs"])


def validate_ui_admission(revision, request):
    return gradio_app.validate_ui_chapter_admission(
        revision,
        "Use latest finalized Traditional Chinese SRT",
        {},
        None,
        "Auto",
        "",
        "key",
        "gpt-4o-mini",
        "",
        "",
        None,
        "",
        None,
        {},
        request=request,
    )[0]


class GradioYoutubeChapterIntegrationTests(unittest.TestCase):
    def test_chapter_queue_and_prequeue_admission_are_bounded(self):
        self.assertEqual(
            gradio_app.demo._queue.max_size,
            gradio_app.CHAPTER_QUEUE_MAX_SIZE,
        )
        self.assertTrue(
            gradio_app.demo._queue.chapter_admission_cleanup_installed
        )
        validators = {
            getattr(block_fn.fn, "__name__", ""): block_fn.validator
            for block_fn in gradio_app.demo.fns.values()
            if block_fn.validator is not None
        }
        self.assertIs(
            validators["api_youtube_chapters"],
            gradio_app.validate_api_chapter_admission,
        )
        self.assertIn(gradio_app.validate_ui_chapter_admission, validators.values())

        edit_binding = next(
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "")
            == "stage_edited_chapter_outputs"
        )
        self.assertEqual(edit_binding.trigger_mode, "always_last")
        self.assertIsNone(edit_binding.validator)
        commit_ids = {
            fn_id
            for fn_id, block_fn in gradio_app.demo.fns.items()
            if getattr(block_fn.fn, "__name__", "")
            in {
                "commit_chapter_generation_outputs",
                "commit_chapter_edit_outputs",
            }
        }
        finalizers = [
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "")
            == "finalize_chapter_publication"
        ]
        self.assertEqual(len(finalizers), 1)
        self.assertTrue(
            all(block_fn.trigger_after in commit_ids for block_fn in finalizers)
        )

    def test_admission_rejects_duplicates_and_releases_every_terminal_path(self):
        ui_request = SimpleNamespace(
            session_hash="chapter-admission-ui",
            client=SimpleNamespace(host="198.51.100.10"),
        )
        api_request = SimpleNamespace(
            session_hash="chapter-admission-api-a",
            client=SimpleNamespace(host="198.51.100.20"),
        )
        rotated_api_request = SimpleNamespace(
            session_hash="chapter-admission-api-b",
            client=SimpleNamespace(host="198.51.100.20"),
        )
        with patch.object(gradio_app, "CHAPTER_ADMISSIONS", OrderedDict()):
            first_ui = validate_ui_admission(0, ui_request)
            duplicate_ui = validate_ui_admission(0, ui_request)
            overlapping_edit = gradio_app.validate_chapter_edit_admission(
                0, "chapters", {}, ui_request
            )
            self.assertTrue(first_ui["is_valid"])
            self.assertFalse(duplicate_ui["is_valid"])
            self.assertFalse(overlapping_edit[0]["is_valid"])

            with gradio_app.chapter_admission_execution(
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION, ui_request
            ):
                pass
            self.assertTrue(
                validate_ui_admission(0, ui_request)[
                    "is_valid"
                ]
            )
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION, ui_request
            )

            api_inputs = (
                SIMPLE_SRT,
                "Auto",
                "",
                "key",
                "gpt-4o-mini",
                "",
                "",
                0,
            )
            first_api = gradio_app.validate_api_chapter_admission(
                *api_inputs, request=api_request
            )
            rotated_api = gradio_app.validate_api_chapter_admission(
                *api_inputs, request=rotated_api_request
            )
            self.assertTrue(all(item["is_valid"] for item in first_api))
            self.assertFalse(rotated_api[0]["is_valid"])

            with gradio_app.chapter_admission_execution(
                gradio_app._CHAPTER_API_GENERATION_ADMISSION,
                api_request,
            ), patch.object(
                gradio_app, "generate_youtube_chapters"
            ) as generate, self.assertRaisesRegex(
                gr.Error, "already pending"
            ):
                gradio_app.api_youtube_chapters(
                    *api_inputs, request=rotated_api_request
                )
            generate.assert_not_called()
            self.assertTrue(
                gradio_app.validate_api_chapter_admission(
                    *api_inputs, request=rotated_api_request
                )[0]["is_valid"]
            )
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_API_GENERATION_ADMISSION,
                rotated_api_request,
            )

            edit_validation = gradio_app.validate_chapter_edit_admission(
                0, "chapters", {}, ui_request
            )
            self.assertTrue(edit_validation[0]["is_valid"])
            self.assertFalse(
                validate_ui_admission(0, ui_request)[
                    "is_valid"
                ]
            )
            edit_key = (
                gradio_app._CHAPTER_EDIT_ADMISSION,
                "session:chapter-admission-ui",
            )
            old_token = gradio_app.CHAPTER_ADMISSIONS[edit_key]["token"]
            gradio_app.CHAPTER_ADMISSIONS[edit_key]["updated_at"] = 0
            self.assertTrue(
                gradio_app.validate_chapter_edit_admission(
                    0, "new chapters", {}, ui_request
                )[0]["is_valid"]
            )
            replacement_token = gradio_app.CHAPTER_ADMISSIONS[edit_key]["token"]
            self.assertNotEqual(replacement_token, old_token)
            self.assertFalse(
                gradio_app.release_chapter_admission(
                    gradio_app._CHAPTER_EDIT_ADMISSION,
                    ui_request,
                    old_token,
                )
            )
            claimed_token = gradio_app._claim_chapter_admission(
                gradio_app._CHAPTER_EDIT_ADMISSION, ui_request
            )
            self.assertEqual(claimed_token, replacement_token)
            self.assertTrue(
                gradio_app.release_chapter_admission(
                    gradio_app._CHAPTER_EDIT_ADMISSION,
                    ui_request,
                    claimed_token,
                )
            )
            self.assertTrue(
                gradio_app.validate_chapter_edit_admission(
                    0, "new chapters", {}, ui_request
                )[0]["is_valid"]
            )
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_EDIT_ADMISSION, ui_request
            )

    def test_queue_cancellation_releases_validator_admission(self):
        request = SimpleNamespace(
            session_hash="chapter-admission-cancelled",
            client=SimpleNamespace(host="198.51.100.40"),
        )
        with patch.object(gradio_app, "CHAPTER_ADMISSIONS", OrderedDict()):
            self.assertTrue(
                validate_ui_admission(0, request)["is_valid"]
            )
            self.assertTrue(gradio_app.CHAPTER_ADMISSIONS)

            asyncio.run(
                gradio_app.demo._queue.clean_events(
                    session_hash=request.session_hash
                )
            )

            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)
            self.assertTrue(
                validate_ui_admission(0, request)["is_valid"]
            )
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION, request
            )

    def test_queue_rejected_publication_lease_expires_exact_candidate(self):
        request = SimpleNamespace(session_hash="chapter-queue-full-publication")
        with patch.multiple(
            gradio_app,
            CHAPTER_ADMISSIONS=OrderedDict(),
            CHAPTER_REVISIONS=OrderedDict(),
            CHAPTER_SESSION_RUNTIME=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS=OrderedDict(),
            STAGED_CHAPTER_OUTPUT_RESERVATIONS=OrderedDict(),
            OWNED_CHAPTER_ARTIFACTS=OrderedDict(),
            CHAPTER_ARTIFACT_RESERVATIONS=0,
            CHAPTER_ADMISSION_TTL_SECONDS=30,
        ):
            self.assertTrue(validate_ui_admission(0, request)["is_valid"])
            revision = gradio_app._get_chapter_request_revision(
                request, gradio_app._CHAPTER_UI_GENERATION_ADMISSION
            )
            token = gradio_app._claim_chapter_admission(
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION, request
            )
            artifact_path = gradio_app.create_owned_chapter_artifact(
                "queue-full", "youtube_chapters_", "00:00 節目導覽"
            )
            result = {
                "valid": True,
                "duration_ms": 35_000,
                "base_name": "queue-full",
                "source": "uploaded SRT",
                "_artifact_path": artifact_path,
            }
            candidate = gradio_app.make_revisioned_output_candidate(
                ("00:00 節目導覽", artifact_path, "valid", result),
                revision,
                request,
                channel="youtube_generation",
            )
            candidate["admission_token"] = token
            self.assertTrue(
                gradio_app.mark_chapter_admission_publishing(
                    gradio_app._CHAPTER_UI_GENERATION_ADMISSION,
                    request,
                    token,
                    candidate,
                )
            )
            next(iter(gradio_app.CHAPTER_ADMISSIONS.values()))[
                "updated_at"
            ] = 0

            self.assertEqual(
                gradio_app.cleanup_expired_chapter_admissions(now=31), 1
            )
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)
            self.assertFalse(gradio_app.STAGED_CHAPTER_OUTPUTS)
            self.assertFalse(Path(artifact_path).exists())
            self.assertTrue(
                validate_ui_admission(revision, request)["is_valid"]
            )
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION, request
            )

    def test_stale_event_cannot_claim_or_release_a_replacement_token(self):
        stale_request = SimpleNamespace(
            session_hash="chapter-admission-exact",
            client=SimpleNamespace(host="198.51.100.41"),
        )
        replacement_request = SimpleNamespace(
            session_hash="chapter-admission-exact",
            client=SimpleNamespace(host="198.51.100.41"),
        )
        with patch.object(gradio_app, "CHAPTER_ADMISSIONS", OrderedDict()):
            self.assertTrue(
                validate_ui_admission(0, stale_request)[
                    "is_valid"
                ]
            )
            key = (
                gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                "session:chapter-admission-exact",
            )
            stale_token = gradio_app.CHAPTER_ADMISSIONS[key]["token"]
            cancellation_snapshot = {(key, stale_token)}
            self.assertTrue(
                gradio_app.release_chapter_admission(
                    gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                    stale_request,
                    stale_token,
                )
            )
            self.assertTrue(
                validate_ui_admission(0, replacement_request)["is_valid"]
            )
            replacement_token = gradio_app.CHAPTER_ADMISSIONS[key]["token"]
            self.assertNotEqual(stale_token, replacement_token)
            self.assertIsNone(
                gradio_app._claim_chapter_admission(
                    gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                    stale_request,
                )
            )
            self.assertEqual(
                gradio_app.release_cancelled_chapter_admissions(
                    cancellation_snapshot
                ),
                0,
            )
            self.assertEqual(
                gradio_app._claim_chapter_admission(
                    gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                    replacement_request,
                ),
                replacement_token,
            )
            self.assertTrue(
                gradio_app.release_chapter_admission(
                    gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                    replacement_request,
                    replacement_token,
                )
            )

    def test_validator_and_execution_share_token_through_raw_request_state(self):
        raw_request = SimpleNamespace(state=SimpleNamespace())
        validator_request = gr.Request(
            request=raw_request,
            session_hash="chapter-real-request-shape",
        )
        execution_request = gr.Request(
            request=raw_request,
            session_hash="chapter-real-request-shape",
        )
        with patch.multiple(
            gradio_app,
            CHAPTER_ADMISSIONS=OrderedDict(),
            CHAPTER_REVISIONS=OrderedDict(),
        ):
            self.assertTrue(
                validate_ui_admission(0, validator_request)["is_valid"]
            )
            token = gradio_app._get_chapter_request_admission_token(
                execution_request,
                gradio_app._CHAPTER_UI_ACTION_ADMISSION,
            )
            self.assertIsNotNone(token)
            self.assertEqual(
                gradio_app._claim_chapter_admission(
                    gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                    execution_request,
                ),
                token,
            )
            self.assertTrue(
                gradio_app.release_chapter_admission(
                    gradio_app._CHAPTER_UI_ACTION_ADMISSION,
                    execution_request,
                    token,
                )
            )

    def test_validator_bound_generation_stages_and_commits_exact_revision(self):
        request = SimpleNamespace(session_hash="chapter-bound-generation")
        generated_outputs = (
            "",
            None,
            "❌ validation failed",
            {
                "valid": False,
                "chapters": [],
                "chapters_text": "",
                "error": "validation failed",
            },
        )
        with patch.multiple(
            gradio_app,
            CHAPTER_ADMISSIONS=OrderedDict(),
            CHAPTER_REVISIONS=OrderedDict(),
            CHAPTER_SESSION_RUNTIME=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS=OrderedDict(),
            STAGED_CHAPTER_OUTPUT_RESERVATIONS=OrderedDict(),
        ), patch.object(
            gradio_app,
            "safe_youtube_chapter_wrapper",
            return_value=generated_outputs,
        ):
            self.assertTrue(validate_ui_admission(0, request)["is_valid"])
            expected_revision = gradio_app._get_chapter_request_revision(
                request,
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION,
            )
            staged = gradio_app.stage_youtube_chapter_outputs(
                0,
                "Use latest finalized Traditional Chinese SRT",
                {},
                None,
                "Auto",
                "",
                "key",
                "gpt-4o-mini",
                "",
                "",
                None,
                "old",
                "/forged/old.txt",
                {"valid": True},
                request,
            )
            self.assertEqual(staged[0], expected_revision)
            self.assertEqual(staged[1:5], gradio_app.reset_chapter_artifacts())
            self.assertEqual(
                gradio_app.commit_chapter_generation_outputs(
                    staged[5], expected_revision, request
                ),
                generated_outputs,
            )
            self.assertEqual(
                next(iter(gradio_app.CHAPTER_ADMISSIONS.values()))["state"],
                "publishing",
            )
            self.assertFalse(
                validate_ui_admission(expected_revision, request)["is_valid"]
            )
            gradio_app.finalize_chapter_publication(staged[5], request)
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)
            self.assertFalse(gradio_app.STAGED_CHAPTER_OUTPUTS)

    def test_api_admission_releases_after_validation_error_and_provider_error(self):
        request = SimpleNamespace(
            session_hash="chapter-admission-api-release",
            client=SimpleNamespace(host="198.51.100.30"),
        )
        api_inputs = (
            SIMPLE_SRT,
            "Auto",
            "",
            "key",
            "gpt-4o-mini",
            "",
            "",
            0,
        )
        with patch.object(gradio_app, "CHAPTER_ADMISSIONS", OrderedDict()):
            self.assertTrue(
                gradio_app.validate_api_chapter_admission(
                    *api_inputs, request=request
                )[0]["is_valid"]
            )
            with self.assertRaisesRegex(gr.Error, "No SRT content"):
                gradio_app.api_youtube_chapters(
                    "", *api_inputs[1:], request=request
                )
            self.assertNotIn(
                (
                    gradio_app._CHAPTER_API_GENERATION_ADMISSION,
                    "client:198.51.100.30",
                ),
                gradio_app.CHAPTER_ADMISSIONS,
            )

            self.assertTrue(
                gradio_app.validate_api_chapter_admission(
                    *api_inputs, request=request
                )[0]["is_valid"]
            )
            with patch.object(
                gradio_app,
                "generate_youtube_chapters",
                side_effect=TimeoutError("provider timeout"),
            ), self.assertRaisesRegex(gr.Error, "network/API timeout"):
                gradio_app.api_youtube_chapters(
                    *api_inputs, request=request
                )
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)

            self.assertTrue(
                gradio_app.validate_api_chapter_admission(
                    *api_inputs, request=request
                )[0]["is_valid"]
            )
            generated = {
                "chapters": [
                    {"start_ms": 0, "title": "節目導覽"},
                    {"start_ms": 12_000, "title": "背景脈絡"},
                    {"start_ms": 24_000, "title": "核心方法"},
                ],
                "chapters_text": (
                    "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法"
                ),
            }
            with patch.object(
                gradio_app,
                "generate_youtube_chapters",
                return_value=generated,
            ):
                result = gradio_app.api_youtube_chapters(
                    *api_inputs, request=request
                )
            self.assertEqual(result["chapters_text"], generated["chapters_text"])
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)

    def test_provider_errors_redact_exact_and_structured_credentials(self):
        request = SimpleNamespace(
            session_hash="chapter-secret-redaction",
            client=SimpleNamespace(host="198.51.100.31"),
        )
        exact_key = "arbitrary-private-key"
        padded_key = f"  {exact_key}  "
        leaked_error = (
            f"transport failed with {exact_key}; "
            "Authorization: Bearer bearer-value-123; "
            "https://user:password@example.test/v1?api_key=query-secret"
        )
        with patch.object(
            gradio_app, "CHAPTER_ADMISSIONS", OrderedDict()
        ), patch.object(
            gradio_app,
            "generate_youtube_chapters",
            side_effect=RuntimeError(leaked_error),
        ):
            inputs = (
                SIMPLE_SRT,
                "Auto",
                "",
                padded_key,
                "gpt-4o-mini",
                "",
                "",
                0,
            )
            self.assertTrue(
                gradio_app.validate_api_chapter_admission(
                    *inputs, request=request
                )[0]["is_valid"]
            )
            with self.assertRaises(gr.Error) as caught:
                gradio_app.api_youtube_chapters(*inputs, request=request)
            public_message = str(caught.exception)
            for secret in (
                exact_key,
                padded_key,
                "bearer-value-123",
                "password",
                "query-secret",
            ):
                self.assertNotIn(secret, public_message)
            self.assertIn("details were hidden", public_message)
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)

        with patch.object(
            gradio_app,
            "run_youtube_chapter_generation",
            side_effect=RuntimeError(leaked_error),
        ):
            _, _, status, result = gradio_app.safe_youtube_chapter_wrapper(
                "source",
                {},
                None,
                "Auto",
                "",
                padded_key,
                "gpt-4o-mini",
                "",
                "",
            )
        self.assertIn("details were hidden", status)
        self.assertFalse(result["valid"])
        for secret in (
            exact_key,
            "bearer-value-123",
            "password",
            "query-secret",
        ):
            self.assertNotIn(secret, status)
            self.assertNotIn(secret, result["error"])

    def test_revision_capacity_never_evicts_a_live_foreign_session(self):
        first_request = SimpleNamespace(session_hash="chapter-capacity-first")
        second_request = SimpleNamespace(session_hash="chapter-capacity-second")
        with patch.multiple(
            gradio_app,
            CHAPTER_REVISIONS=OrderedDict(),
            CHAPTER_SESSION_RUNTIME=OrderedDict(),
            CHAPTER_REVISION_REGISTRY_LIMIT=1,
        ):
            first_revision = gradio_app.advance_chapter_revision(0, first_request)
            artifact_path = gradio_app.create_owned_chapter_artifact(
                "capacity", "youtube_chapters_", "00:00 節目導覽"
            )
            outputs = (
                "00:00 節目導覽",
                artifact_path,
                "valid",
                {
                    "valid": True,
                    "duration_ms": 35_000,
                    "base_name": "capacity",
                    "source": "latest Traditional SRT",
                },
            )
            self.assertTrue(
                gradio_app._record_authoritative_chapter_runtime(
                    first_request.session_hash,
                    first_revision,
                    "youtube_generation",
                    outputs,
                )
            )

            with self.assertRaisesRegex(gr.Error, "Too many active chapter sessions"):
                gradio_app.advance_chapter_revision(0, second_request)

            self.assertIn(
                first_request.session_hash, gradio_app.CHAPTER_SESSION_RUNTIME
            )
            self.assertTrue(Path(artifact_path).exists())
            self.assertNotIn(second_request.session_hash, gradio_app.CHAPTER_REVISIONS)

            gradio_app.CHAPTER_SESSION_RUNTIME[first_request.session_hash][
                "updated_at"
            ] = time.time() - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS - 1
            second_revision = gradio_app.advance_chapter_revision(0, second_request)

            self.assertEqual(second_revision, 1)
            self.assertNotIn(
                first_request.session_hash, gradio_app.CHAPTER_SESSION_RUNTIME
            )
            self.assertFalse(Path(artifact_path).exists())

    def test_staged_capacity_preserves_current_foreign_result_and_rejects_new_work(self):
        first_request = SimpleNamespace(session_hash="staged-capacity-first")
        second_request = SimpleNamespace(session_hash="staged-capacity-second")
        with patch.multiple(
            gradio_app,
            CHAPTER_REVISIONS=OrderedDict(),
            CHAPTER_SESSION_RUNTIME=OrderedDict(),
            CHAPTER_SESSION_UPLOADS=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS=OrderedDict(),
            STAGED_CHAPTER_OUTPUT_RESERVATIONS=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS_LIMIT=1,
            CHAPTER_REVISION_REGISTRY_LIMIT=2,
            CHAPTER_ADMISSIONS=OrderedDict(),
            OWNED_CHAPTER_ARTIFACTS=OrderedDict(),
            CHAPTER_ARTIFACT_RESERVATIONS=0,
        ):
            first_revision = gradio_app.advance_chapter_revision(
                0, first_request
            )
            second_revision = gradio_app.advance_chapter_revision(
                0, second_request
            )
            first_path = gradio_app.create_owned_chapter_artifact(
                "first", "youtube_chapters_", "00:00 第一個結果"
            )
            first_outputs = (
                "00:00 第一個結果",
                first_path,
                "valid",
                {"valid": False},
            )
            gradio_app.make_revisioned_output_candidate(
                first_outputs,
                first_revision,
                first_request,
                channel="youtube_generation",
            )

            self.assertTrue(
                validate_ui_admission(
                    second_revision, second_request
                )["is_valid"]
            )
            with patch.object(
                gradio_app, "safe_youtube_chapter_wrapper"
            ) as generation, self.assertRaisesRegex(
                gr.Error, "waiting to publish"
            ):
                gradio_app.stage_youtube_chapter_outputs(
                    second_revision,
                    "Use latest finalized Traditional Chinese SRT",
                    {},
                    None,
                    "Auto",
                    "",
                    "key",
                    "gpt-4o-mini",
                    "",
                    "",
                    None,
                    "",
                    None,
                    {},
                    second_request,
                )
            generation.assert_not_called()
            self.assertTrue(Path(first_path).exists())
            self.assertIn(
                (
                    first_request.session_hash,
                    "youtube_generation",
                    first_revision,
                ),
                gradio_app.STAGED_CHAPTER_OUTPUTS,
            )
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)
            self.assertTrue(
                gradio_app.remove_owned_chapter_artifact(first_path)
            )

    def test_revision_capacity_treats_current_staged_result_as_live(self):
        first_request = SimpleNamespace(session_hash="staged-revision-first")
        second_request = SimpleNamespace(session_hash="staged-revision-second")
        with patch.multiple(
            gradio_app,
            CHAPTER_REVISIONS=OrderedDict(),
            CHAPTER_SESSION_RUNTIME=OrderedDict(),
            CHAPTER_SESSION_UPLOADS=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS=OrderedDict(),
            STAGED_CHAPTER_OUTPUT_RESERVATIONS=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS_LIMIT=1,
            CHAPTER_REVISION_REGISTRY_LIMIT=1,
            OWNED_CHAPTER_ARTIFACTS=OrderedDict(),
            CHAPTER_ARTIFACT_RESERVATIONS=0,
        ):
            first_revision = gradio_app.advance_chapter_revision(
                0, first_request
            )
            first_path = gradio_app.create_owned_chapter_artifact(
                "first-live", "youtube_chapters_", "00:00 第一個結果"
            )
            gradio_app.make_revisioned_output_candidate(
                ("00:00 第一個結果", first_path, "valid", {"valid": False}),
                first_revision,
                first_request,
                channel="youtube_generation",
            )
            with self.assertRaisesRegex(
                gr.Error, "Too many active chapter sessions"
            ):
                gradio_app.advance_chapter_revision(0, second_request)

            self.assertTrue(Path(first_path).exists())
            self.assertIn(first_request.session_hash, gradio_app.CHAPTER_REVISIONS)
            self.assertNotIn(
                second_request.session_hash, gradio_app.CHAPTER_REVISIONS
            )
            gradio_app.OWNED_CHAPTER_ARTIFACTS[first_path] = (
                time.time() - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS - 1
            )
            self.assertEqual(
                gradio_app.cleanup_owned_chapter_artifacts(),
                0,
            )
            self.assertTrue(Path(first_path).exists())
            staged_key = (
                first_request.session_hash,
                "youtube_generation",
                first_revision,
            )
            gradio_app.STAGED_CHAPTER_OUTPUTS[staged_key] = (
                time.time() - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS - 1,
                gradio_app.STAGED_CHAPTER_OUTPUTS[staged_key][1],
            )
            self.assertEqual(gradio_app.cleanup_owned_chapter_artifacts(), 1)
            self.assertFalse(Path(first_path).exists())
            self.assertNotIn(staged_key, gradio_app.STAGED_CHAPTER_OUTPUTS)

    def test_model_identifier_is_bounded_and_validated_before_generation(self):
        valid_models = [
            "gpt-4o-mini",
            "anthropic/claude-3-haiku-20240307",
            "openrouter/meta-llama/llama-3.1-8b",
            "bedrock/us.amazon.nova-lite-v1:0",
        ]
        for model in valid_models:
            with self.subTest(model=model):
                self.assertEqual(
                    gradio_app.validate_llm_model_identifier(model), model
                )

        invalid_models = [
            None,
            123,
            "",
            " gpt-4o-mini",
            "gpt 4o",
            "gpt\x00model",
            "gpt\u202emodel",
            "gpt🎬model",
            "ｇｐｔ-model",
            "https://provider.example/model",
            "m" * (gradio_app.MAX_LLM_MODEL_IDENTIFIER_CHARACTERS + 1),
        ]
        for model in invalid_models:
            with self.subTest(model=repr(model)), self.assertRaises(gr.Error):
                gradio_app.validate_llm_model_identifier(model)

        with patch.object(gradio_app, "generate_youtube_chapters") as generate:
            with self.assertRaises(gr.Error):
                gradio_app.api_youtube_chapters(
                    SIMPLE_SRT,
                    "Auto",
                    "",
                    "caller-key",
                    "bad\nmodel",
                    "",
                    "",
                )
        generate.assert_not_called()

    def test_custom_base_url_never_receives_the_server_api_key(self):
        environment = {
            "OPENAI_API_KEY": "server-secret",
            "OPENAI_BASE_URL": "https://trusted.example/v1",
            "FUNCLIP_TRUSTED_LLM_MODELS": "",
        }
        with patch.dict(os.environ, environment, clear=False):
            with self.assertRaisesRegex(gr.Error, "custom Base URL requires its own API key"):
                gradio_app.resolve_llm_config(
                    "", "gpt-4o-mini", "", "https://attacker.example/v1"
                )

            key, base_url, model = gradio_app.resolve_llm_config(
                "", "gpt-4o-mini", "", environment["OPENAI_BASE_URL"]
            )
            self.assertEqual((key, base_url, model), (
                environment["OPENAI_API_KEY"],
                environment["OPENAI_BASE_URL"],
                "gpt-4o-mini",
            ))

            key, base_url, _ = gradio_app.resolve_llm_config(
                "caller-key", "gpt-4o-mini", "", "https://caller.example/v1"
            )
            self.assertEqual((key, base_url), ("caller-key", "https://caller.example/v1"))

    def test_server_openai_key_never_reaches_user_selected_provider(self):
        environment = {
            "OPENAI_API_KEY": "server-secret",
            "OPENAI_BASE_URL": "",
            "FUNCLIP_TRUSTED_LLM_MODELS": "",
        }
        with patch.dict(os.environ, environment, clear=False):
            provider_selections = [
                ("gemini-1.5-flash", ""),
                ("Custom", "anthropic/claude-3-haiku-20240307"),
                ("Custom", "claude-3-haiku-20240307"),
                ("Custom", "gpt-fake/anthropic-model"),
                ("Custom", "command-r"),
                ("Custom", "jamba-1.5-large"),
                ("Custom", "amazon.titan-text-express-v1"),
            ]
            for model_name, custom_model in provider_selections:
                with self.subTest(model=model_name, custom=custom_model), self.assertRaisesRegex(
                    gr.Error, "requires its own API key"
                ):
                    gradio_app.resolve_llm_config(
                        "", model_name, custom_model, ""
                    )

            key, _, model = gradio_app.resolve_llm_config(
                "caller-gemini-key", "gemini-1.5-flash", "", ""
            )
            self.assertEqual((key, model), ("caller-gemini-key", "gemini-1.5-flash"))

    def test_trusted_gateway_requires_an_exact_model_alias_allowlist(self):
        environment = {
            "OPENAI_API_KEY": "server-secret",
            "OPENAI_BASE_URL": "https://trusted.example/v1",
            "FUNCLIP_TRUSTED_LLM_MODELS": "house-model, command-r",
        }
        with patch.dict(os.environ, environment, clear=False):
            key, base_url, model = gradio_app.resolve_llm_config(
                "", "Custom", "house-model", environment["OPENAI_BASE_URL"]
            )
            self.assertEqual(
                (key, base_url, model),
                ("server-secret", environment["OPENAI_BASE_URL"], "house-model"),
            )

            key, _, model = gradio_app.resolve_llm_config(
                "", "Custom", "COMMAND-R", ""
            )
            self.assertEqual((key, model), ("server-secret", "COMMAND-R"))

            for alias in ("house-model-v2", "jamba-1.5-large"):
                with self.subTest(alias=alias), self.assertRaisesRegex(
                    gr.Error, "requires its own API key"
                ):
                    gradio_app.resolve_llm_config(
                        "", "Custom", alias, environment["OPENAI_BASE_URL"]
                    )

    def test_single_then_batch_translation_invalidates_singular_latest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.srt"
            second = Path(directory) / "second.srt"
            empty = Path(directory) / "empty.srt"
            first.write_text(SIMPLE_SRT, encoding="utf-8")
            second.write_text(SIMPLE_SRT.replace("字幕", "內容"), encoding="utf-8")
            empty.write_text("", encoding="utf-8")

            single_result = gradio_app.safe_translate_traditional_wrapper(
                [str(first)], {"keep": True}
            )
            single_state = single_result[4]
            self.assertIn("latest_traditional_srt", single_state)
            self.assertTrue(single_state["keep"])

            batch_result = gradio_app.safe_translate_traditional_wrapper(
                [str(first), str(second)], single_state
            )
            batch_state = batch_result[4]
            self.assertNotIn("latest_traditional_srt", batch_state)
            self.assertNotIn("latest_traditional_base_name", batch_state)
            self.assertTrue(batch_state["keep"])

            partial_batch_result = gradio_app.safe_translate_traditional_wrapper(
                [str(first), str(empty)], single_state
            )
            partial_batch_state = partial_batch_result[4]
            self.assertNotIn("latest_traditional_srt", partial_batch_state)
            self.assertTrue(partial_batch_state["keep"])

            missing = Path(directory) / "missing.srt"
            failed_batch_result = gradio_app.safe_translate_traditional_wrapper(
                [str(first), str(missing)], single_state
            )
            failed_batch_state = failed_batch_result[4]
            self.assertNotIn("latest_traditional_srt", failed_batch_state)
            self.assertTrue(failed_batch_state["keep"])

            all_empty_result = gradio_app.safe_translate_traditional_wrapper(
                [str(empty), str(empty)], single_state
            )
            all_empty_state = all_empty_result[4]
            self.assertNotIn("latest_traditional_srt", all_empty_state)
            self.assertTrue(all_empty_state["keep"])

    def test_non_chinese_or_malformed_translation_invalidates_latest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            english = Path(directory) / "english.srt"
            malformed = Path(directory) / "malformed.srt"
            english.write_text(
                "1\n00:00:00,000 --> 00:00:35,000\nEnglish subtitles only\n",
                encoding="utf-8",
            )
            malformed.write_text("這不是有效的 SRT", encoding="utf-8")
            for path in (english, malformed):
                with self.subTest(path=path.name):
                    result = gradio_app.safe_translate_traditional_wrapper(
                        [str(path)],
                        {"latest_traditional_srt": "old", "keep": True},
                    )
                    state = result[4]
                    self.assertNotIn("latest_traditional_srt", state)
                    self.assertTrue(state["keep"])

    def test_failed_single_file_producers_invalidate_latest_source(self):
        translator_callback = next(
            block_fn.fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_translator_correction_outputs"
        )
        transcription_callback = next(
            block_fn.fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_correction_outputs"
        )
        old_state = {
            "original_srt": SIMPLE_SRT,
            "latest_traditional_srt": "old",
            "keep": True,
        }
        translation_result = gradio_app.safe_translate_traditional_wrapper(
            ["/definitely/missing/single.srt"], old_state
        )
        self.assertNotIn("latest_traditional_srt", translation_result[4])

        with patch.object(
            gradio_app,
            "run_llm_correction_for_files",
            side_effect=RuntimeError("single correction failed"),
        ):
            translator_result = translator_callback(
                "Upload SRT file(s)",
                ["single.srt"],
                {},
                old_state,
                "key",
                "gpt-4o-mini",
                "",
                "",
                None,
            )
        translator_result = staged_outputs(translator_result)
        self.assertNotIn("latest_traditional_srt", translator_result[-1])

        with patch.object(
            gradio_app,
            "run_llm_correction_for_content",
            side_effect=RuntimeError("transcription correction failed"),
        ):
            transcription_result = transcription_callback(
                "key",
                "gpt-4o-mini",
                "",
                "",
                old_state,
                "Use output from previous step",
                None,
                None,
            )
        transcription_result = staged_outputs(transcription_result)
        self.assertNotIn("latest_traditional_srt", transcription_result[-1])
        self.assertTrue(transcription_result[-1]["keep"])

    def test_uploaded_utf8_bom_is_read_and_invalid_encoding_fails_clearly(self):
        request = SimpleNamespace(session_hash="chapter-local-upload")
        foreign_request = SimpleNamespace(session_hash="chapter-foreign-upload")
        def upload_event(name, raw_content):
            return gr.EventData(None, {
                "name": name,
                "data_base64": gradio_app.base64.b64encode(raw_content).decode("ascii"),
            })

        with patch.object(
            gradio_app, "CHAPTER_REVISIONS", OrderedDict()
        ), patch.object(
            gradio_app, "CHAPTER_SESSION_RUNTIME", OrderedDict()
        ), patch.object(
            gradio_app, "CHAPTER_SESSION_UPLOADS", OrderedDict()
        ):
            revision, *_ = gradio_app.begin_chapter_upload_revision(
                0,
                event=upload_event(
                    "valid.srt", ("\ufeff" + SIMPLE_SRT).encode("utf-8")
                ),
                request=request,
            )
            content, base_name, source = gradio_app.resolve_chapter_srt(
                "Upload Chinese SRT",
                {},
                "/forged/browser/path.srt",
                request,
                revision,
            )
            self.assertTrue(content.startswith("\ufeff"))
            self.assertEqual((base_name, source), ("valid", "uploaded SRT"))

            revision, *_ = gradio_app.begin_chapter_upload_revision(
                revision,
                event=upload_event(
                    "double-bom.srt",
                    b"\xef\xbb\xbf\xef\xbb\xbf" + SIMPLE_SRT.encode("utf-8"),
                ),
                request=request,
            )
            with patch(
                "funclip.llm.youtube_chapters.completion"
            ) as completion, self.assertRaisesRegex(
                gradio_app.ChapterGenerationError, "three usable"
            ):
                gradio_app.run_youtube_chapter_generation(
                    "Upload Chinese SRT",
                    {},
                    "/forged/browser/path.srt",
                    "Auto",
                    "",
                    "key",
                    "gpt-4o-mini",
                    "",
                    "",
                    request=request,
                    expected_chapter_revision=revision,
                )
            completion.assert_not_called()

            invalid_revision, _, _, invalid_status, invalid_result = (
                gradio_app.begin_chapter_upload_revision(
                    revision,
                    event=upload_event("invalid.srt", b"\xff\xfe\xfa"),
                    request=request,
                )
            )
            self.assertGreater(invalid_revision, revision)
            self.assertIn("UTF-8 encoded", invalid_status)
            self.assertIn("previous uploaded SRT was cleared", invalid_status)
            self.assertFalse(invalid_result["valid"])
            with self.assertRaisesRegex(gr.Error, "session-bound"):
                gradio_app.resolve_chapter_srt(
                    "Upload Chinese SRT",
                    {},
                    "/forged/browser/path.srt",
                    request,
                    invalid_revision,
                )

            foreign_revision = gradio_app.advance_chapter_revision(
                0, foreign_request
            )
            with self.assertRaisesRegex(gr.Error, "session-bound"):
                gradio_app.resolve_chapter_srt(
                    "Upload Chinese SRT",
                    {},
                    "/known/foreign/cache/path.srt",
                    foreign_request,
                    foreign_revision,
                )

    def test_long_upload_stem_is_bounded_before_artifact_creation(self):
        long_stem = "a" * 240
        event = gr.EventData(None, {
            "name": f"{long_stem}.srt",
            "data_base64": gradio_app.base64.b64encode(
                SIMPLE_SRT.encode("utf-8")
            ).decode("ascii"),
        })
        _, base_name = gradio_app._decode_local_chapter_upload_event(event)
        self.assertLessEqual(
            len(base_name.encode("utf-8")),
            gradio_app.MAX_SAFE_ARTIFACT_BASE_BYTES,
        )
        output_path = gradio_app.create_owned_chapter_artifact(
            base_name,
            "youtube_chapters_edited_",
            "00:00 節目導覽",
        )
        try:
            self.assertLessEqual(
                len(Path(output_path).name.encode("utf-8")),
                gradio_app.PORTABLE_FILENAME_COMPONENT_BYTES,
            )
        finally:
            gradio_app.remove_owned_chapter_artifact(output_path)

    def test_oversized_upload_is_rejected_before_reading_or_generation(self):
        event = gr.EventData(None, {
            "name": "oversized.srt",
            "data_base64": gradio_app.base64.b64encode(
                SIMPLE_SRT.encode("utf-8")
            ).decode("ascii"),
        })
        with patch.object(
            gradio_app, "MAX_SRT_UTF8_BYTES", 1
        ), patch("builtins.open") as open_file:
            _, _, _, status, result = gradio_app.begin_chapter_upload_revision(
                0,
                event=event,
                request=SimpleNamespace(session_hash="chapter-oversized-upload"),
            )
        self.assertIn("UTF-8 limit", status)
        self.assertFalse(result["valid"])
        open_file.assert_not_called()

    def test_rapid_editor_inputs_coalesce_without_rejecting_latest_text(self):
        request = SimpleNamespace(session_hash="youtube-chapter-rapid-edit")
        text_a = "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法"
        text_b = "00:00 最新導覽\n00:15 最新背景\n00:30 最新方法"
        with patch.multiple(
            gradio_app,
            CHAPTER_REVISIONS=OrderedDict(),
            CHAPTER_SESSION_RUNTIME=OrderedDict(),
            STAGED_CHAPTER_OUTPUTS=OrderedDict(),
            STAGED_CHAPTER_OUTPUT_RESERVATIONS=OrderedDict(),
            OWNED_CHAPTER_ARTIFACTS=OrderedDict(),
            CHAPTER_ARTIFACT_RESERVATIONS=0,
        ):
            source_revision = gradio_app.advance_chapter_revision(0, request)
            source_path = gradio_app.create_owned_chapter_artifact(
                "rapid", "youtube_chapters_", text_a
            )
            source_result = {
                "valid": True,
                "duration_ms": 60_000,
                "base_name": "rapid",
                "source": "latest Traditional SRT",
                "_artifact_path": source_path,
            }
            self.assertTrue(
                gradio_app._record_authoritative_chapter_runtime(
                    request.session_hash,
                    source_revision,
                    "youtube_generation",
                    (text_a, source_path, "valid", source_result),
                )
            )

            staged_a = gradio_app.stage_edited_chapter_outputs(
                source_revision, text_a, source_result, request
            )
            staged_b = gradio_app.stage_edited_chapter_outputs(
                staged_a[0], text_b, staged_a[3], request
            )
            stale_commit = gradio_app.commit_chapter_edit_outputs(
                staged_a[4], staged_a[0], request
            )
            self.assertTrue(
                all(value.get("__type__") == "update" for value in stale_commit)
            )
            committed_b = gradio_app.commit_chapter_edit_outputs(
                staged_b[4], staged_b[0], request
            )
            self.assertEqual(committed_b[2]["chapters_text"], text_b)
            self.assertEqual(
                Path(committed_b[0]).read_text(encoding="utf-8"), text_b
            )
            self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)
            gradio_app.remove_owned_chapter_artifact(committed_b[2])

    def test_edited_text_synchronizes_validation_result_and_download_artifact(self):
        previous = {"duration_ms": 48_000, "base_name": "video", "valid": True}
        valid_text = "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法"
        request = SimpleNamespace(session_hash="youtube-chapter-edit-start")
        source_revision = gradio_app.advance_chapter_revision(0, request)
        source_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", valid_text
        )
        source_outputs = (
            valid_text,
            source_path,
            "valid",
            {
                **previous,
                "source": "latest Traditional SRT",
                "_artifact_path": source_path,
            },
        )
        source_candidate = gradio_app.make_revisioned_output_candidate(
            source_outputs,
            source_revision,
            request,
            channel="youtube_generation",
        )
        self.assertEqual(
            gradio_app.commit_chapter_generation_outputs(
                source_candidate, source_revision, request
            ),
            source_outputs,
        )
        self.assertTrue(
            gradio_app.validate_chapter_edit_admission(
                source_revision,
                valid_text,
                previous,
                request,
            )[0]["is_valid"]
        )
        admission_token = gradio_app._get_chapter_request_admission_token(
            request,
            gradio_app._CHAPTER_EDIT_ADMISSION,
        )
        revision, download, status, pending = gradio_app.begin_chapter_edit_revision(
            source_revision,
            valid_text,
            previous,
            request,
        )
        self.assertTrue(
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_EDIT_ADMISSION,
                request,
                admission_token,
            )
        )
        self.assertGreater(revision, 0)
        self.assertIsNone(download)
        self.assertIn("Validating", status)
        self.assertEqual(pending["duration_ms"], 48_000)
        self.assertFalse(pending["valid"])
        self.assertEqual(pending["chapters"], [])

        path, status, result = gradio_app.sync_edited_chapters(valid_text, previous)
        first_path = path
        self.assertEqual(Path(path).read_bytes(), valid_text.encode("utf-8"))
        self.assertEqual(result["chapters_text"], valid_text)
        self.assertTrue(result["valid"])
        self.assertIn("valid for YouTube", status)

        structural_text = (
            "00:00 第一集劇情分析\n"
            "00:12 第二課核心方法\n"
            "00:24 第三篇實作流程"
        )
        path, status, result = gradio_app.sync_edited_chapters(
            structural_text, result
        )
        self.assertFalse(Path(first_path).exists())
        self.assertEqual(Path(path).read_bytes(), structural_text.encode("utf-8"))
        self.assertTrue(result["valid"])
        self.assertIn("valid for YouTube", status)

        invalid_text = "00:00 節目導覽\n00:05 太近"
        structural_path = path
        path, status, result = gradio_app.sync_edited_chapters(invalid_text, result)
        self.assertIsNone(path)
        self.assertFalse(Path(structural_path).exists())
        self.assertFalse(result["valid"])
        self.assertEqual(result["chapters"], [])
        self.assertIn("not valid for YouTube", status)

        malformed_spacing = [
            " 00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法",
            "00:00 節目導覽 \n00:12 背景脈絡\n00:24 核心方法",
            "00:00  節目導覽\n00:12 背景脈絡\n00:24 核心方法",
            "00:00 節目導覽\n\n00:12 背景脈絡\n00:24 核心方法",
            "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法\n",
            "00:00 節目導覽\r\n00:12 背景脈絡\r\n00:24 核心方法",
            "060:00 節目導覽\n060:12 背景脈絡\n060:24 核心方法",
            "00:00 節目導覽\n00:12 背景脈絡\n01:00:00 核心方法",
            "00:00 [核心方法][id]\n00:12 背景脈絡\n00:24 實作建議",
            "00:00 核心分析1000:00\n00:12 背景脈絡\n00:24 實作建議",
            "00:00 節目導覽\n00:12 核心方法\n00:24 核心方法介紹",
            "00:00 台灣美食文化\n00:12 臺灣美食文化\n00:24 核心方法建議",
            "00:00 吃飯健康指南\n00:12 喫飯健康指南\n00:24 核心方法建議",
            "00:00 節目導覽\n" + ("9" * 5_000) + ":00 背景脈絡\n00:24 核心方法",
            "00:00 第一單元\n00:12 第二單元\n00:24 第三單元",
            "00:00 第一集\n00:12 第二課\n00:24 第三篇",
            "00:00 部分一\n00:12 集數二\n00:24 課程三",
            "00:00 節目導覽\n00:12 相機第Ⅰ代設定\n00:24 相機第一代設定",
            "00:00 節目導覽\n00:12 相機第I代設定\n00:24 相機第一代設定",
            "00:00 結論\n00:12 摘要\n00:24 簡介",
            "00:00 話說回來\n00:12 順帶一提\n00:24 換句話說",
            "00:00 課堂一\n00:12 課堂二\n00:24 課堂三",
            "00:00 方法法法\n00:12 技術術術\n00:24 流程程程",
            "00:00 節目導覽\n00:12 iPhone2 相機設定\n00:24 iPhone二 相機設定",
            "00:00 節目導覽\n00:12 iPhone2 相機設定\n00:24 iPhoneⅡ 相機設定",
            "00:00 節目導覽\n00:12 相機第 II 代設定\n00:24 相機第二代設定",
            "00:00 節目導覽\n00:12 第一課核心方法\n00:24 第I課核心方法",
            "00:00 節目導覽\n00:12 相機第ↅ代設定\n00:24 相機第六代設定",
        ]
        for text in malformed_spacing:
            with self.subTest(text=text):
                path, status, result = gradio_app.sync_edited_chapters(text, previous)
                self.assertIsNone(path)
                self.assertFalse(result["valid"])
                self.assertEqual(result["chapters"], [])
                self.assertIn("not valid for YouTube", status)

        invalid_duration_states = [
            True,
            False,
            float("nan"),
            float("inf"),
            10**100,
            gradio_app.MAX_VIDEO_DURATION_MS + 1,
        ]
        for duration_ms in invalid_duration_states:
            with self.subTest(duration_ms=duration_ms):
                path, status, result = gradio_app.sync_edited_chapters(
                    valid_text,
                    {"duration_ms": duration_ms, "base_name": "video", "valid": True},
                )
                self.assertIsNone(path)
                self.assertFalse(result["valid"])
                self.assertEqual(result["chapters"], [])
                self.assertIn("not valid for YouTube", status)

        thirteen_hour_text = (
            "00:00 節目導覽\n06:00:00 背景脈絡\n13:00:00 核心方法"
        )
        path, status, result = gradio_app.sync_edited_chapters(
            thirteen_hour_text,
            {
                "duration_ms": 13 * 60 * 60 * 1_000 + 10_000,
                "base_name": "video",
                "valid": True,
            },
        )
        self.assertIsNone(path)
        self.assertFalse(result["valid"])
        self.assertIn("not valid for YouTube", status)

        with patch.object(
            gradio_app,
            "create_owned_chapter_artifact",
            side_effect=RuntimeError("temporary artifact capacity unavailable"),
        ):
            path, status, result = gradio_app.sync_edited_chapters(valid_text, previous)
        self.assertIsNone(path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["chapters"], [])
        self.assertIn("Unable to create", status)

    def test_generated_download_exactly_matches_visible_chapter_text(self):
        chapters_text = "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法"
        generated = {
            "chapters_text": chapters_text,
            "chapters": [
                {"start_ms": 0, "title": "節目導覽"},
                {"start_ms": 12_000, "title": "背景脈絡"},
                {"start_ms": 24_000, "title": "核心方法"},
            ],
            "duration_ms": 48_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "chapters.txt"
            with patch.object(
                gradio_app,
                "resolve_chapter_srt",
                return_value=(
                    "台灣吃飯郁達夫憂鬱症",
                    "video",
                    "latest Traditional SRT",
                ),
            ), patch.object(
                gradio_app,
                "resolve_llm_config",
                return_value=("key", "", "model"),
            ), patch.object(
                gradio_app,
                "generate_youtube_chapters",
                return_value=generated.copy(),
            ) as generate_chapters, patch.object(
                gradio_app,
                "create_unique_text_path",
                return_value=str(output_path),
            ):
                visible, download, _, result = gradio_app.run_youtube_chapter_generation(
                    "Use latest Traditional SRT",
                    {"latest_traditional_video_duration_ms": 60_000},
                    None,
                    "Auto",
                    "",
                    "key",
                    "model",
                    "",
                    "",
                )

            self.assertEqual(visible, chapters_text)
            self.assertEqual(download, str(output_path))
            self.assertEqual(result["chapters_text"], visible)
            self.assertEqual(output_path.read_bytes(), visible.encode("utf-8"))
            self.assertEqual(
                generate_chapters.call_args.args[0],
                "台灣吃飯郁達夫憂鬱症",
            )
            self.assertIs(
                generate_chapters.call_args.kwargs["title_transform"],
                gradio_app.normalize_traditional_for_validation,
            )
            self.assertEqual(
                generate_chapters.call_args.kwargs["video_duration_ms"], 60_000
            )
            self.assertTrue(gradio_app.remove_owned_chapter_artifact(result))
            self.assertFalse(output_path.exists())

    def test_uploaded_srt_ignores_default_zero_video_duration(self):
        chapters_text = "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法"
        generated = {
            "chapters_text": chapters_text,
            "chapters": [
                {"start_ms": 0, "title": "節目導覽"},
                {"start_ms": 12_000, "title": "背景脈絡"},
                {"start_ms": 24_000, "title": "核心方法"},
            ],
            "duration_ms": 48_000,
        }
        with patch.object(
            gradio_app,
            "resolve_chapter_srt",
            return_value=(SIMPLE_SRT, "video", "uploaded SRT"),
        ), patch.object(
            gradio_app,
            "resolve_llm_config",
            return_value=("key", "", "model"),
        ), patch.object(
            gradio_app,
            "generate_youtube_chapters",
            return_value=generated.copy(),
        ) as generate_chapters, patch.object(
            gradio_app,
            "create_owned_chapter_artifact",
            return_value="/tmp/video_youtube_chapters.txt",
        ):
            _, _, status, _ = gradio_app.run_youtube_chapter_generation(
                "Upload Chinese SRT",
                {},
                "/tmp/video.srt",
                "Auto",
                "",
                "key",
                "model",
                "",
                "",
                0.0,
            )

        self.assertIsNone(
            generate_chapters.call_args.kwargs["video_duration_ms"]
        )
        self.assertIn("SRT timeline extent", status)

    def test_uploaded_simplified_srt_is_converted_before_generation(self):
        generated = {
            "chapters_text": "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法",
            "chapters": [
                {"start_ms": 0, "title": "節目導覽"},
                {"start_ms": 12_000, "title": "背景脈絡"},
                {"start_ms": 24_000, "title": "核心方法"},
            ],
            "duration_ms": 48_000,
        }
        with patch.object(
            gradio_app,
            "resolve_chapter_srt",
            return_value=(SIMPLE_SRT, "video", "uploaded SRT"),
        ), patch.object(
            gradio_app,
            "resolve_llm_config",
            return_value=("key", "", "model"),
        ), patch.object(
            gradio_app,
            "generate_youtube_chapters",
            return_value=generated.copy(),
        ) as generate_chapters, patch.object(
            gradio_app,
            "create_owned_chapter_artifact",
            return_value="/tmp/video_youtube_chapters.txt",
        ):
            _, _, status, result = gradio_app.run_youtube_chapter_generation(
                "Upload Chinese SRT",
                {},
                "/tmp/video.srt",
                "Auto",
                "",
                "key",
                "model",
                "",
                "",
                0.0,
            )

        self.assertIn("簡體中文字幕", generate_chapters.call_args.args[0])
        self.assertNotIn("简体中文字幕", generate_chapters.call_args.args[0])
        self.assertTrue(result["input_converted_to_traditional"])
        self.assertIn("automatically converted", status)

    def test_served_gradio_download_is_the_exact_owned_artifact(self):
        output_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", "00:00 節目導覽"
        )
        component = gr.File()
        served_path = asyncio.run(
            component.async_move_resource_to_block_cache(output_path)
        )

        self.assertEqual(served_path, output_path)
        self.assertEqual(component.GRADIO_CACHE, gradio_app.CHAPTER_ARTIFACT_CACHE_DIR)
        self.assertIn(output_path, component.temp_files)
        self.assertIn(output_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertTrue(gradio_app.remove_owned_chapter_artifact(served_path))
        self.assertFalse(Path(served_path).exists())

    def test_abandoned_chapter_artifact_expires_without_another_generation(self):
        output_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", "00:00 節目導覽"
        )
        with gradio_app.OWNED_CHAPTER_ARTIFACTS_LOCK:
            gradio_app.OWNED_CHAPTER_ARTIFACTS[output_path] = (
                gradio_app.time.time()
                - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS
                - 1
            )
        self.assertEqual(gradio_app.cleanup_owned_chapter_artifacts(), 1)
        self.assertFalse(Path(output_path).exists())

    def test_expired_artifact_cleanup_retries_after_delete_failure(self):
        output_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", "00:00 節目導覽"
        )
        with gradio_app.OWNED_CHAPTER_ARTIFACTS_LOCK:
            original_created_at = (
                gradio_app.time.time()
                - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS
                - 1
            )
            gradio_app.OWNED_CHAPTER_ARTIFACTS[output_path] = original_created_at

        with patch.object(gradio_app.os, "remove", side_effect=PermissionError):
            self.assertEqual(gradio_app.cleanup_owned_chapter_artifacts(), 0)
        with gradio_app.OWNED_CHAPTER_ARTIFACTS_LOCK:
            self.assertEqual(
                gradio_app.OWNED_CHAPTER_ARTIFACTS[output_path], original_created_at
            )
        self.assertTrue(Path(output_path).exists())

        self.assertEqual(gradio_app.cleanup_owned_chapter_artifacts(), 1)
        self.assertNotIn(output_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertFalse(Path(output_path).exists())

    def test_forced_cleanup_retries_after_delete_failure(self):
        output_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", "00:00 節目導覽"
        )
        with patch.object(gradio_app.os, "remove", side_effect=PermissionError):
            self.assertEqual(
                gradio_app.cleanup_owned_chapter_artifacts(force=True), 0
            )
        self.assertIn(output_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertTrue(Path(output_path).exists())

        self.assertEqual(gradio_app.cleanup_owned_chapter_artifacts(force=True), 1)
        self.assertNotIn(output_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertFalse(Path(output_path).exists())

    def test_capacity_preserves_fresh_artifact_then_reclaims_expired(self):
        first_path = gradio_app.create_owned_chapter_artifact(
            "first", "youtube_chapters_", "00:00 第一段落"
        )
        with patch.object(
            gradio_app, "CHAPTER_ARTIFACT_REGISTRY_LIMIT", 1
        ), patch.object(gradio_app, "create_unique_text_path") as create_path:
            with self.assertRaisesRegex(RuntimeError, "still active"):
                gradio_app.create_owned_chapter_artifact(
                    "second", "youtube_chapters_", "00:00 第二段落"
                )
            create_path.assert_not_called()

        self.assertIn(first_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertTrue(Path(first_path).exists())
        self.assertEqual(gradio_app.CHAPTER_ARTIFACT_RESERVATIONS, 0)

        with gradio_app.OWNED_CHAPTER_ARTIFACTS_LOCK:
            gradio_app.OWNED_CHAPTER_ARTIFACTS[first_path] = (
                time.time() - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS - 1
            )
        with patch.object(
            gradio_app, "CHAPTER_ARTIFACT_REGISTRY_LIMIT", 1
        ), patch.object(
            gradio_app.os, "remove", side_effect=PermissionError
        ), self.assertRaisesRegex(RuntimeError, "still active"):
            gradio_app.create_owned_chapter_artifact(
                "second", "youtube_chapters_", "00:00 第二段落"
            )
        self.assertIn(first_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)

        with patch.object(gradio_app, "CHAPTER_ARTIFACT_REGISTRY_LIMIT", 1):
            second_path = gradio_app.create_owned_chapter_artifact(
                "second", "youtube_chapters_", "00:00 第二段落"
            )
        self.assertFalse(Path(first_path).exists())
        self.assertNotIn(first_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertIn(second_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertTrue(gradio_app.remove_owned_chapter_artifact(second_path))

    def test_capacity_never_deletes_a_live_foreign_runtime_artifact(self):
        request = SimpleNamespace(session_hash="artifact-capacity-live")
        first_path = gradio_app.create_owned_chapter_artifact(
            "first", "youtube_chapters_", "00:00 第一段落"
        )
        revisions = OrderedDict([(request.session_hash, 1)])
        runtimes = OrderedDict()
        with patch.object(
            gradio_app, "CHAPTER_REVISIONS", revisions
        ), patch.object(
            gradio_app, "CHAPTER_SESSION_RUNTIME", runtimes
        ):
            self.assertTrue(
                gradio_app._record_authoritative_chapter_runtime(
                    request.session_hash,
                    1,
                    "youtube_generation",
                    (
                        "00:00 第一段落",
                        first_path,
                        "valid",
                        {
                            "valid": True,
                            "duration_ms": 35_000,
                            "base_name": "first",
                            "source": "latest Traditional SRT",
                        },
                    ),
                )
            )
            with gradio_app.OWNED_CHAPTER_ARTIFACTS_LOCK:
                gradio_app.OWNED_CHAPTER_ARTIFACTS[first_path] = (
                    time.time() - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS - 1
                )
            self.assertEqual(gradio_app.cleanup_owned_chapter_artifacts(), 0)
            with patch.object(
                gradio_app, "CHAPTER_ARTIFACT_REGISTRY_LIMIT", 1
            ), self.assertRaisesRegex(RuntimeError, "still active"):
                gradio_app.create_owned_chapter_artifact(
                    "second", "youtube_chapters_", "00:00 第二段落"
                )
            self.assertEqual(runtimes[request.session_hash]["artifact_path"], first_path)
            self.assertTrue(Path(first_path).exists())
            self.assertIn(first_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)
        self.assertTrue(gradio_app.remove_owned_chapter_artifact(first_path))

    def test_cleanup_and_authoritative_publication_are_atomic(self):
        request = SimpleNamespace(session_hash="artifact-cleanup-race")
        artifact_path = gradio_app.create_owned_chapter_artifact(
            "race", "youtube_chapters_", "00:00 競態測試"
        )
        with gradio_app.OWNED_CHAPTER_ARTIFACTS_LOCK:
            gradio_app.OWNED_CHAPTER_ARTIFACTS[artifact_path] = (
                time.time() - gradio_app.CHAPTER_ARTIFACT_TTL_SECONDS - 1
            )
        revisions = OrderedDict([(request.session_hash, 1)])
        runtimes = OrderedDict()
        delete_started = threading.Event()
        allow_delete = threading.Event()
        record_finished = threading.Event()
        cleanup_result = []
        record_result = []
        original_delete = gradio_app._delete_owned_chapter_artifact_locked

        def delayed_delete(path):
            delete_started.set()
            self.assertTrue(allow_delete.wait(2))
            return original_delete(path)

        def run_cleanup():
            cleanup_result.append(gradio_app.cleanup_owned_chapter_artifacts())

        def run_record():
            record_result.append(
                gradio_app._record_authoritative_chapter_runtime(
                    request.session_hash,
                    1,
                    "youtube_generation",
                    (
                        "00:00 競態測試",
                        artifact_path,
                        "valid",
                        {
                            "valid": True,
                            "duration_ms": 35_000,
                            "base_name": "race",
                            "source": "latest Traditional SRT",
                        },
                    ),
                )
            )
            record_finished.set()

        with patch.object(
            gradio_app, "CHAPTER_REVISIONS", revisions
        ), patch.object(
            gradio_app, "CHAPTER_SESSION_RUNTIME", runtimes
        ), patch.object(
            gradio_app,
            "_delete_owned_chapter_artifact_locked",
            side_effect=delayed_delete,
        ):
            cleanup_thread = threading.Thread(target=run_cleanup)
            record_thread = threading.Thread(target=run_record)
            cleanup_thread.start()
            self.assertTrue(delete_started.wait(2))
            record_thread.start()
            self.assertFalse(record_finished.wait(0.05))
            allow_delete.set()
            cleanup_thread.join(2)
            record_thread.join(2)

        self.assertEqual(cleanup_result, [1])
        self.assertEqual(record_result, [False])
        self.assertFalse(Path(artifact_path).exists())
        self.assertNotIn(request.session_hash, runtimes)

    def test_failed_regeneration_clears_prior_chapter_artifacts(self):
        successful = (
            "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法",
            "/tmp/old-chapters.txt",
            "✅ valid",
            {"valid": True, "chapters": [1, 2, 3]},
        )
        with patch.object(
            gradio_app, "run_youtube_chapter_generation", return_value=successful
        ):
            self.assertEqual(
                gradio_app.safe_youtube_chapter_wrapper(
                    "source", {}, None, "Auto", "", "key", "model", "", ""
                )[0],
                successful[0],
            )

        with patch.object(
            gradio_app,
            "run_youtube_chapter_generation",
            side_effect=gr.Error("new source failed"),
        ):
            output, download, status, result = gradio_app.safe_youtube_chapter_wrapper(
                "source", {}, None, "Auto", "", "key", "model", "", ""
            )
        self.assertEqual(output, "")
        self.assertIsNone(download)
        self.assertIn("new source failed", status)
        self.assertFalse(result["valid"])
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["chapters_text"], "")

    def test_producer_attempts_are_wired_to_clear_populated_chapter_artifacts(self):
        populated = (
            "00:00 舊的章節",
            "/tmp/old-chapters.txt",
            {"valid": True, "chapters": [{"start_ms": 0}], "chapters_text": "old"},
        )
        output, download, status, result = gradio_app.reset_chapter_artifacts(*populated)
        self.assertEqual(output, "")
        self.assertIsNone(download)
        self.assertIn("source changed", status.lower())
        self.assertFalse(result["valid"])
        self.assertEqual(result["chapters"], [])
        self.assertEqual(result["chapters_text"], "")

        old_state = {"latest_traditional_srt": "old", "keep": True}
        cleared_state, output, download, status, result = (
            gradio_app.reset_chapter_source_and_artifacts(old_state, *populated)
        )
        self.assertNotIn("latest_traditional_srt", cleared_state)
        self.assertTrue(cleared_state["keep"])
        self.assertEqual(output, "")
        self.assertIsNone(download)
        self.assertFalse(result["valid"])

        with self.assertWarnsRegex(UserWarning, "upload a file"):
            no_file_result = gradio_app.process_media(None, old_state)
        self.assertNotIn("latest_traditional_srt", no_file_result[-1])

        artifact_reset_bindings = [
            (fn_id, block_fn)
            for fn_id, block_fn in gradio_app.demo.fns.items()
            if getattr(block_fn.fn, "__name__", "") == "begin_chapter_artifact_revision"
        ]
        source_reset_bindings = [
            (fn_id, block_fn)
            for fn_id, block_fn in gradio_app.demo.fns.items()
            if getattr(block_fn.fn, "__name__", "") == "begin_chapter_source_revision"
        ]
        upload_bindings = [
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "")
            == "begin_chapter_upload_revision"
        ]
        self.assertEqual(len(artifact_reset_bindings), 8)
        self.assertEqual(len(upload_bindings), 1)
        self.assertEqual(len(source_reset_bindings), 10)
        self.assertTrue(all(len(block_fn.inputs) == 4 for _, block_fn in artifact_reset_bindings))
        self.assertTrue(all(len(block_fn.outputs) == 5 for _, block_fn in artifact_reset_bindings))
        self.assertTrue(all(len(block_fn.inputs) == 5 for _, block_fn in source_reset_bindings))
        self.assertTrue(all(len(block_fn.outputs) == 6 for _, block_fn in source_reset_bindings))
        producer_callbacks = [
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") in {
                "stage_process_media_outputs",
                "stage_correction_outputs",
                "stage_traditional_translation_outputs",
                "stage_translator_correction_outputs",
            }
        ]
        source_reset_ids = {fn_id for fn_id, _ in source_reset_bindings}
        self.assertEqual(len({block_fn.trigger_after for block_fn in producer_callbacks}), 4)
        self.assertTrue(all(block_fn.trigger_after in source_reset_ids for block_fn in producer_callbacks))

        generation_callback = next(
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_youtube_chapter_outputs"
        )
        self.assertIsNone(generation_callback.trigger_after)

        self.assertTrue(all(len(block_fn.inputs) >= 3 for block_fn in producer_callbacks))
        self.assertEqual(len(generation_callback.inputs), 14)
        self.assertTrue(all(len(block_fn.outputs) == 1 for block_fn in producer_callbacks))
        self.assertEqual(len(generation_callback.outputs), 6)

        producer_setting_revision_bindings = [
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "advance_chapter_revision"
        ]
        self.assertEqual(len(producer_setting_revision_bindings), 12)
        self.assertTrue(all(
            len(block_fn.inputs) == 1 and len(block_fn.outputs) == 1
            for block_fn in producer_setting_revision_bindings
        ))
        producer_setting_labels = [
            gradio_app.demo.blocks[block_fn.targets[0][0]].label
            for block_fn in producer_setting_revision_bindings
        ]
        self.assertCountEqual(
            producer_setting_labels,
            [
                "API Key (OpenAI/Compatible)",
                "Model",
                "Custom Model Name",
                "Base URL (Optional)",
                "API Key",
                "Model",
                "Custom Model Name",
                "Base URL (Optional)",
                "API Key",
                "Model",
                "Custom Model Name",
                "Base URL (Optional)",
            ],
        )
        artifact_trigger_labels = [
            gradio_app.demo.blocks[block_fn.targets[0][0]].label
            for _, block_fn in artifact_reset_bindings
            if block_fn.targets and block_fn.targets[0][0] is not None
        ]
        self.assertIn("API Key", artifact_trigger_labels)

        english_revision_binding = next(
            (fn_id, block_fn)
            for fn_id, block_fn in gradio_app.demo.fns.items()
            if getattr(block_fn.fn, "__name__", "") == "begin_english_translation_revision"
        )
        self.assertEqual(
            (len(english_revision_binding[1].inputs), len(english_revision_binding[1].outputs)),
            (1, 1),
        )
        english_callback = next(
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_english_translation_outputs"
        )
        self.assertEqual(english_callback.trigger_after, english_revision_binding[0])
        self.assertEqual(len(english_callback.inputs), 6)

        publication_callback_names = {
            "advance_chapter_revision",
            "begin_english_translation_revision",
            "begin_chapter_artifact_revision",
            "begin_chapter_upload_revision",
            "begin_chapter_source_revision",
            "begin_chapter_edit_revision",
            "commit_process_outputs",
            "commit_correction_outputs",
            "commit_traditional_translation_outputs",
            "commit_english_translation_outputs",
            "commit_translator_correction_outputs",
            "commit_chapter_generation_outputs",
            "commit_chapter_edit_outputs",
        }
        publication_callbacks = [
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") in publication_callback_names
        ]
        self.assertTrue(publication_callbacks)
        self.assertEqual(
            {
                getattr(block_fn.fn, "__name__", "")
                for block_fn in publication_callbacks
                if getattr(block_fn.fn, "__name__", "").startswith("commit_")
            },
            {name for name in publication_callback_names if name.startswith("commit_")},
        )
        self.assertTrue(all(
            block_fn.concurrency_id
            == gradio_app.CHAPTER_PUBLICATION_CONCURRENCY_ID
            and block_fn.concurrency_limit == 1
            for block_fn in publication_callbacks
        ))

    def test_late_generation_and_producer_results_cannot_overwrite_newer_state(self):
        request = SimpleNamespace(session_hash="youtube-chapter-race-test")
        first_revision = gradio_app.advance_chapter_revision(0, request)
        second_revision = gradio_app.advance_chapter_revision(0, request)
        self.assertGreater(second_revision, first_revision)

        stale_no_file = gradio_app.process_media(
            None,
            {"latest_traditional_srt": "newer", "keep": True},
            first_revision,
            request=request,
        )
        self.assertTrue(all(value.get("__type__") == "update" for value in stale_no_file))

        generated = (
            "00:00 舊章節\n00:12 舊背景\n00:24 舊方法",
            "/tmp/nonexistent-stale-chapters.txt",
            "✅ stale",
            {"valid": True},
        )
        with patch.object(
            gradio_app, "run_youtube_chapter_generation", return_value=generated
        ):
            stale_generation = gradio_app.safe_youtube_chapter_wrapper(
                "source",
                {},
                None,
                "Auto",
                "",
                "key",
                "model",
                "",
                "",
                None,
                first_revision,
                request,
            )
        self.assertTrue(all(value.get("__type__") == "update" for value in stale_generation))

        translated_state = {
            "latest_output_srt": SIMPLE_SRT,
            "latest_output_base_name": "old",
        }
        with patch.object(
            gradio_app,
            "translate_srt_to_traditional",
            return_value=("old", "old", "/tmp/old.srt", translated_state),
        ):
            stale_producer = gradio_app.safe_translate_traditional_wrapper(
                ["old.srt"], {}, first_revision, request
            )
        self.assertTrue(all(value.get("__type__") == "update" for value in stale_producer))

        english_state = {
            "latest_output_paths": ["/tmp/old-english.srt"],
            "latest_output_kind": "english",
        }
        with patch.object(
            gradio_app,
            "translate_srt_to_english_fn",
            return_value=("old", "old", "/tmp/old-english.srt", english_state),
        ):
            stale_english = gradio_app.safe_translate_english_wrapper(
                ["old.srt"],
                "key",
                "gpt-4o-mini",
                "",
                "",
                first_revision,
                request,
            )
        self.assertTrue(all(value.get("__type__") == "update" for value in stale_english))

        stale_edit = gradio_app.sync_edited_chapters(
            "00:00 節目導覽\n00:12 背景脈絡\n00:24 核心方法",
            {"duration_ms": 48_000},
            first_revision,
            request,
        )
        self.assertTrue(all(value.get("__type__") == "update" for value in stale_edit))

    def test_queued_stale_generation_never_reaches_paid_generation(self):
        request = SimpleNamespace(session_hash="chapter-stale-before-paid")
        first_revision = gradio_app.advance_chapter_revision(0, request)
        gradio_app.advance_chapter_revision(first_revision, request)

        with patch.object(
            gradio_app, "run_youtube_chapter_generation"
        ) as generate:
            self.assertTrue(
                validate_ui_admission(
                    first_revision, request
                )["is_valid"]
            )
            queued_revision = gradio_app._get_chapter_request_revision(
                request,
                gradio_app._CHAPTER_UI_GENERATION_ADMISSION,
            )
            gradio_app.advance_chapter_revision(queued_revision, request)
            with self.assertRaisesRegex(gr.Error, "superseded"):
                gradio_app.stage_youtube_chapter_outputs(
                    first_revision,
                    "Use latest finalized Traditional SRT",
                    {},
                    None,
                    "Auto",
                    "",
                    "key",
                    "gpt-4o-mini",
                    "",
                    "",
                    None,
                    "",
                    None,
                    {},
                    request,
                )

        generate.assert_not_called()
        self.assertFalse(gradio_app.CHAPTER_ADMISSIONS)

    def test_revisioned_commit_rejects_result_superseded_after_final_check(self):
        request = SimpleNamespace(session_hash="chapter-commit-race")
        first_revision = gradio_app.advance_chapter_revision(0, request)
        artifact_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", "00:00 節目導覽"
        )
        outputs = (
            "00:00 節目導覽",
            artifact_path,
            "valid",
            {"valid": True, "_artifact_path": artifact_path},
        )
        candidate = gradio_app.make_revisioned_output_candidate(
            outputs,
            first_revision,
            request,
            channel="youtube_generation",
        )

        second_revision = gradio_app.advance_chapter_revision(
            first_revision, request
        )
        committed = gradio_app.commit_chapter_generation_outputs(
            candidate, second_revision, request
        )

        self.assertTrue(
            all(value.get("__type__") == "update" for value in committed)
        )
        self.assertFalse(Path(artifact_path).exists())
        self.assertNotIn(artifact_path, gradio_app.OWNED_CHAPTER_ARTIFACTS)

    def test_late_staging_cannot_overwrite_current_revision_candidate(self):
        request = SimpleNamespace(session_hash="chapter-stage-race")
        first_revision = gradio_app.advance_chapter_revision(0, request)
        second_revision = gradio_app.advance_chapter_revision(
            first_revision, request
        )
        current_path = gradio_app.create_owned_chapter_artifact(
            "current", "youtube_chapters_", "00:00 目前章節"
        )
        current_outputs = (
            "00:00 目前章節",
            current_path,
            "current",
            {
                "valid": True,
                "duration_ms": 48_000,
                "base_name": "current",
                "source": "latest Traditional SRT",
                "_artifact_path": current_path,
            },
        )
        gradio_app.make_revisioned_output_candidate(
            current_outputs,
            second_revision,
            request,
            channel="youtube_generation",
        )

        stale_path = gradio_app.create_owned_chapter_artifact(
            "stale", "youtube_chapters_", "00:00 過時章節"
        )
        stale_trigger = gradio_app.make_revisioned_output_candidate(
            (
                "00:00 過時章節",
                stale_path,
                "stale",
                {"valid": True, "_artifact_path": stale_path},
            ),
            first_revision,
            request,
            channel="youtube_generation",
        )
        committed = gradio_app.commit_chapter_generation_outputs(
            stale_trigger, second_revision, request
        )

        self.assertEqual(committed, current_outputs)
        self.assertFalse(Path(stale_path).exists())
        self.assertTrue(Path(current_path).exists())
        self.assertTrue(gradio_app.remove_owned_chapter_artifact(current_path))

    def test_editor_uses_server_duration_and_cannot_delete_foreign_artifact(self):
        request = SimpleNamespace(session_hash="authoritative-editor")
        revision = gradio_app.advance_chapter_revision(0, request)
        generated_path = gradio_app.create_owned_chapter_artifact(
            "video", "youtube_chapters_", "00:00 原始章節"
        )
        generated_outputs = (
            "00:00 原始章節",
            generated_path,
            "valid",
            {
                "valid": True,
                "duration_ms": 35_000,
                "base_name": "video",
                "source": "latest Traditional SRT",
                "_artifact_path": generated_path,
            },
        )
        candidate = gradio_app.make_revisioned_output_candidate(
            generated_outputs,
            revision,
            request,
            channel="youtube_generation",
        )
        self.assertEqual(
            gradio_app.commit_chapter_generation_outputs(
                candidate, revision, request
            ),
            generated_outputs,
        )

        foreign_path = gradio_app.create_owned_chapter_artifact(
            "foreign", "youtube_chapters_", "00:00 其他工作階段"
        )
        forged_result = {
            "duration_ms": 40_000,
            "base_name": "forged",
            "_artifact_path": foreign_path,
        }
        edited_text = (
            "00:00 節目導覽\n00:10 背景脈絡\n00:30 核心方法"
        )
        self.assertTrue(
            gradio_app.validate_chapter_edit_admission(
                revision,
                edited_text,
                forged_result,
                request,
            )[0]["is_valid"]
        )
        admission_token = gradio_app._get_chapter_request_admission_token(
            request,
            gradio_app._CHAPTER_EDIT_ADMISSION,
        )
        next_revision, _, _, pending = gradio_app.begin_chapter_edit_revision(
            revision,
            edited_text,
            forged_result,
            request,
        )
        self.assertTrue(
            gradio_app.release_chapter_admission(
                gradio_app._CHAPTER_EDIT_ADMISSION,
                request,
                admission_token,
            )
        )
        self.assertEqual(pending["duration_ms"], 35_000)
        self.assertEqual(pending["base_name"], "video")
        self.assertFalse(Path(generated_path).exists())
        self.assertTrue(Path(foreign_path).exists())

        path, status, result = gradio_app.sync_edited_chapters(
            edited_text,
            forged_result,
            next_revision,
            request,
        )
        self.assertIsNone(path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["duration_ms"], 35_000)
        self.assertIn("not valid for YouTube", status)
        self.assertTrue(Path(foreign_path).exists())
        self.assertTrue(gradio_app.remove_owned_chapter_artifact(foreign_path))

    def test_translator_correction_updates_or_invalidates_latest_source(self):
        callback = next(
            block_fn.fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_translator_correction_outputs"
        )
        common = (
            "original", "corrected", "diff", "orig-path", "corr-path", "done"
        )
        with patch.object(
            gradio_app,
            "run_llm_correction_for_files",
            return_value=(*common, "最新繁體字幕", "corrected_traditional"),
        ):
            result = callback(
                "Upload SRT file(s)",
                ["one.srt"],
                {},
                {"latest_traditional_srt": "old", "keep": True},
                "key",
                "gpt-4o-mini",
                "",
                "",
                None,
            )
        result = staged_outputs(result)
        single_state = result[-1]
        self.assertEqual(single_state["latest_traditional_srt"], "最新繁體字幕")
        self.assertEqual(single_state["latest_traditional_base_name"], "corrected_traditional")
        self.assertTrue(single_state["keep"])

        with patch.object(
            gradio_app,
            "run_llm_correction_for_files",
            return_value=(*common, "", ""),
        ):
            result = callback(
                "Upload SRT file(s)",
                ["one.srt", "two.srt"],
                {},
                single_state,
                "key",
                "gpt-4o-mini",
                "",
                "",
                None,
            )
        result = staged_outputs(result)
        batch_state = result[-1]
        self.assertNotIn("latest_traditional_srt", batch_state)
        self.assertTrue(batch_state["keep"])

    def test_failed_multi_file_translator_correction_invalidates_latest_source(self):
        callback = next(
            block_fn.fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_translator_correction_outputs"
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.srt"
            second = Path(directory) / "second.srt"
            first.write_text(SIMPLE_SRT, encoding="utf-8")
            second.write_text(SIMPLE_SRT.replace("字幕", "內容"), encoding="utf-8")
            calls = 0

            def correct_then_fail(srt_content, **_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("second correction failed")
                return srt_content

            with patch.object(
                gradio_app, "correct_srt_content", side_effect=correct_then_fail
            ):
                result = callback(
                    "Upload SRT file(s)",
                    [str(first), str(second)],
                    {},
                    {"latest_traditional_srt": "old", "keep": True},
                    "key",
                    "gpt-4o-mini",
                    "",
                    "",
                    None,
                )
            result = staged_outputs(result)
        self.assertEqual(calls, 2)
        state = result[-1]
        self.assertNotIn("latest_traditional_srt", state)
        self.assertTrue(state["keep"])

    def test_correction_file_results_expose_only_a_singular_latest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.srt"
            second = Path(directory) / "second.srt"
            english = Path(directory) / "english.srt"
            empty = Path(directory) / "empty.srt"
            first.write_text(SIMPLE_SRT, encoding="utf-8")
            second.write_text(SIMPLE_SRT.replace("字幕", "內容"), encoding="utf-8")
            english.write_text(
                "1\n00:00:00,000 --> 00:00:35,000\nEnglish subtitles only\n",
                encoding="utf-8",
            )
            empty.write_text("", encoding="utf-8")
            with patch.object(
                gradio_app,
                "correct_srt_content",
                side_effect=lambda srt_content, **_: srt_content,
            ):
                single = gradio_app.run_llm_correction_for_files(
                    [str(first)], "key", "gpt-4o-mini", "", ""
                )
                batch = gradio_app.run_llm_correction_for_files(
                    [str(first), str(second)], "key", "gpt-4o-mini", "", ""
                )
                english_result = gradio_app.run_llm_correction_for_files(
                    [str(english)], "key", "gpt-4o-mini", "", ""
                )
                partial_batch = gradio_app.run_llm_correction_for_files(
                    [str(first), str(empty)], "key", "gpt-4o-mini", "", ""
                )
        self.assertIn("簡體中文字幕", single[-2])
        self.assertTrue(single[-1].endswith("_traditional"))
        self.assertEqual(batch[-2:], ("", ""))
        self.assertEqual(english_result[-2:], ("", ""))
        self.assertEqual(partial_batch[-2:], ("", ""))

    def test_transcription_correction_only_finalizes_valid_chinese_srt(self):
        callback = next(
            block_fn.fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_correction_outputs"
        )
        outputs = [
            "1\n00:00:00,000 --> 00:00:35,000\nEnglish subtitles only\n",
            "這不是有效的 SRT",
        ]
        for finalized_output in outputs:
            correction_result = (
                "original",
                "corrected",
                finalized_output,
                "diff",
                "orig-path",
                "corr-path",
                "trad-path",
                "done",
            )
            with (
                self.subTest(finalized_output=finalized_output),
                patch.object(
                    gradio_app,
                    "run_llm_correction_for_content",
                    return_value=correction_result,
                ),
            ):
                result = callback(
                    "key",
                    "gpt-4o-mini",
                    "",
                    "",
                    {
                        "original_srt": SIMPLE_SRT,
                        "latest_traditional_srt": "old",
                        "keep": True,
                    },
                    "Use output from previous step",
                    None,
                    None,
                )
            result = staged_outputs(result)
            state = result[-1]
            self.assertNotIn("latest_traditional_srt", state)
            self.assertTrue(state["keep"])

        valid_result = (
            SIMPLE_SRT,
            SIMPLE_SRT,
            SIMPLE_SRT,
            "diff",
            "orig-path",
            "corr-path",
            "trad-path",
            "done",
        )
        with patch.object(
            gradio_app,
            "run_llm_correction_for_content",
            return_value=valid_result,
        ):
            result = callback(
                "key",
                "gpt-4o-mini",
                "",
                "",
                {
                    "original_srt": SIMPLE_SRT,
                    "media_duration_seconds": 60.0,
                },
                "Use output from previous step",
                None,
                None,
            )
        result = staged_outputs(result)
        self.assertEqual(
            result[-1]["latest_traditional_video_duration_ms"], 60_000
        )

    def test_corrected_english_translator_output_invalidates_latest_traditional_source(self):
        callback = next(
            block_fn.fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_translator_correction_outputs"
        )
        with tempfile.TemporaryDirectory() as directory:
            english = Path(directory) / "english.srt"
            english.write_text(
                "1\n00:00:00,000 --> 00:00:35,000\nEnglish subtitles only\n",
                encoding="utf-8",
            )
            translator_state = {
                "latest_output_paths": [str(english)],
                "latest_output_kind": "english",
            }
            with patch.object(
                gradio_app,
                "correct_srt_content",
                side_effect=lambda srt_content, **_: srt_content,
            ):
                result = callback(
                    "Use output from previous step",
                    None,
                    translator_state,
                    {"latest_traditional_srt": "stale", "keep": True},
                    "key",
                    "gpt-4o-mini",
                    "",
                    "",
                    None,
                )
        result = staged_outputs(result)
        updated_state = result[-1]
        self.assertNotIn("latest_traditional_srt", updated_state)
        self.assertTrue(updated_state["keep"])

    def test_api_payloads_include_chapters_and_traditional_srt_text(self):
        generated = {
            "chapters": [{"start_ms": 0, "title": "節目導覽"}],
            "chapters_text": "00:00 節目導覽",
            "duration_ms": 35_000,
            "model": "test-model",
        }
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "server-key"}, clear=False),
            patch.object(gradio_app, "generate_youtube_chapters", return_value=generated) as generate,
        ):
            response = gradio_app.api_youtube_chapters(
                SIMPLE_SRT, "Auto", "主題", "", "gpt-4o-mini", "", ""
            )
        self.assertEqual(response["chapters_text"], "00:00 節目導覽")
        self.assertIn("status", response)
        self.assertEqual(generate.call_args.kwargs["density"], "Auto")
        self.assertIsNone(generate.call_args.kwargs["video_duration_ms"])
        self.assertIn("簡體中文字幕", generate.call_args.args[0])
        self.assertTrue(response["input_converted_to_traditional"])

        correction_result = (
            "original", "corrected", "traditional", "diff", "orig-path",
            "corr-path", "trad-path", "done",
        )
        with patch.object(
            gradio_app, "run_llm_correction_for_content", return_value=correction_result
        ):
            response = gradio_app.api_srt_correct(
                SIMPLE_SRT, "key", "gpt-4o-mini", "", "", True
            )
            response_without_traditional = gradio_app.api_srt_correct(
                SIMPLE_SRT, "key", "gpt-4o-mini", "", "", False
            )
        self.assertEqual(response["corrected_traditional_srt"], "traditional")
        self.assertEqual(response["corrected_traditional_file_path"], "trad-path")
        self.assertNotIn("corrected_traditional_srt", response_without_traditional)

    def test_public_duration_inputs_reject_non_finite_and_over_limit_values(self):
        invalid_values = [
            True,
            False,
            "nan",
            "inf",
            "-inf",
            "1e309",
            -1,
            gradio_app.MAX_VIDEO_DURATION_MS / 1000 + 1,
            10 ** 10_000,
        ]
        with (
            patch.object(
                gradio_app,
                "resolve_chapter_srt",
                return_value=(SIMPLE_SRT, "video", "uploaded SRT"),
            ),
            patch.object(
                gradio_app,
                "resolve_llm_config",
                return_value=("key", "", "gpt-4o-mini"),
            ),
            patch.object(gradio_app, "generate_youtube_chapters") as generate,
        ):
            for value in invalid_values:
                with self.subTest(path="ui", value_type=type(value).__name__), self.assertRaises(
                    gr.Error
                ):
                    gradio_app.run_youtube_chapter_generation(
                        "Upload Chinese SRT",
                        {},
                        None,
                        "Auto",
                        "",
                        "key",
                        "gpt-4o-mini",
                        "",
                        "",
                        value,
                    )
                with self.subTest(path="api", value_type=type(value).__name__), self.assertRaises(
                    gr.Error
                ):
                    gradio_app.api_youtube_chapters(
                        SIMPLE_SRT,
                        "Auto",
                        "",
                        "key",
                        "gpt-4o-mini",
                        "",
                        "",
                        value,
                    )
            generate.assert_not_called()

        self.assertEqual(
            gradio_app.parse_video_duration_seconds(12 * 60 * 60),
            gradio_app.MAX_VIDEO_DURATION_MS,
        )
        for value in (0, -1):
            with self.subTest(path="ui-zero-negative", value=value), self.assertRaises(
                gr.Error
            ):
                gradio_app.parse_video_duration_seconds(value)

    def test_gradio_endpoint_and_edit_callback_are_registered(self):
        app_source = Path(gradio_app.__file__).read_text(encoding="utf-8")
        self.assertNotIn("allowed_paths=", app_source)
        api_info = gradio_app.demo.get_api_info()
        endpoint = api_info.get("named_endpoints", {}).get("/youtube_chapters")
        self.assertIsNotNone(endpoint)
        self.assertEqual(len(endpoint["parameters"]), 8)
        self.assertEqual(len(endpoint["returns"]), 1)
        self.assertTrue(any(
            getattr(block_fn.fn, "__name__", "") == "stage_edited_chapter_outputs"
            for block_fn in gradio_app.demo.fns.values()
        ))
        edit_callback = next(
            block_fn
            for block_fn in gradio_app.demo.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "stage_edited_chapter_outputs"
        )
        self.assertEqual(
            (len(edit_callback.inputs), len(edit_callback.outputs)),
            (3, 5),
        )
        editor = next(
            block
            for block in gradio_app.demo.blocks.values()
            if isinstance(block, gr.Code) and block.label == "YouTube Chapters (editable)"
        )
        self.assertEqual(editor.buttons, ["copy"])


if __name__ == "__main__":
    unittest.main()
