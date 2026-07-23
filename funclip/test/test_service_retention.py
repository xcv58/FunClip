import tempfile
import unittest
from pathlib import Path

from funclip.service_retention import (
    MediaCleanupError,
    UploadPathError,
    cleanup_media_files,
    run_with_media_cleanup,
    validate_gradio_upload_path,
)


class SimulatedCancellation(BaseException):
    pass


class ServiceRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.upload_root = self.root / "gradio"
        self.upload_root.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_media_pair(self):
        cache_directory = self.upload_root / "content-digest"
        cache_directory.mkdir()
        cached = cache_directory / "source.wav"
        cached.write_bytes(b"cached media")
        staged = self.root / "funclip_job_source.wav"
        staged.write_bytes(b"staged media")
        return staged, cached

    def test_success_removes_stage_upload_and_empty_cache_directory(self):
        staged, cached = self.make_media_pair()

        result = run_with_media_cleanup(
            lambda: "complete",
            staged_path=staged,
            cached_upload_path=cached,
            upload_root=self.upload_root,
        )

        self.assertEqual(result, "complete")
        self.assertFalse(staged.exists())
        self.assertFalse(cached.exists())
        self.assertFalse(cached.parent.exists())

    def test_failure_is_rethrown_after_cleanup(self):
        staged, cached = self.make_media_pair()

        def fail():
            raise RuntimeError("processing failed")

        with self.assertRaisesRegex(RuntimeError, "processing failed"):
            run_with_media_cleanup(
                fail,
                staged_path=staged,
                cached_upload_path=cached,
                upload_root=self.upload_root,
            )

        self.assertFalse(staged.exists())
        self.assertFalse(cached.exists())

    def test_cancellation_is_rethrown_after_cleanup(self):
        staged, cached = self.make_media_pair()

        def cancel():
            raise SimulatedCancellation()

        with self.assertRaises(SimulatedCancellation):
            run_with_media_cleanup(
                cancel,
                staged_path=staged,
                cached_upload_path=cached,
                upload_root=self.upload_root,
            )

        self.assertFalse(staged.exists())
        self.assertFalse(cached.exists())

    def test_outside_upload_is_never_deleted(self):
        staged = self.root / "funclip_job_source.wav"
        staged.write_bytes(b"staged media")
        outside = self.root / "outside.wav"
        outside.write_bytes(b"outside media")

        with self.assertRaises(MediaCleanupError):
            cleanup_media_files(
                staged,
                outside,
                upload_root=self.upload_root,
            )

        self.assertFalse(staged.exists())
        self.assertTrue(outside.exists())

    def test_symlink_upload_is_rejected_without_deleting_target(self):
        target = self.root / "target.wav"
        target.write_bytes(b"target")
        link = self.upload_root / "linked.wav"
        link.symlink_to(target)

        with self.assertRaises(UploadPathError):
            validate_gradio_upload_path(link, upload_root=self.upload_root)

        self.assertTrue(target.exists())

    def test_missing_cache_file_is_already_clean(self):
        staged = self.root / "funclip_job_source.wav"
        staged.write_bytes(b"staged media")
        missing = self.upload_root / "content-digest" / "missing.wav"

        cleanup_media_files(
            staged,
            missing,
            upload_root=self.upload_root,
        )

        self.assertFalse(staged.exists())

    def test_nonempty_shared_cache_directory_is_retained(self):
        staged, cached = self.make_media_pair()
        sibling = cached.parent / "other.wav"
        sibling.write_bytes(b"other media")

        cleanup_media_files(
            staged,
            cached,
            upload_root=self.upload_root,
        )

        self.assertFalse(cached.exists())
        self.assertTrue(sibling.exists())
        self.assertTrue(sibling.parent.exists())


if __name__ == "__main__":
    unittest.main()
