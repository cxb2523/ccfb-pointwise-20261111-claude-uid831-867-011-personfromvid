"""Unit tests for blur-aware frame selection (FrameSelectorConfig.blur_threshold).

Uses numpy synthetic images: solid-color frames count as blurry (Laplacian
variance ~0), noise frames count as sharp. Because INTER_AREA normalization
averages out high-frequency noise, tests only assert relative ordering and
selection behavior — never absolute blur score values.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

import personfromvid.analysis.frame_selector as frame_selector_module
from personfromvid.analysis.frame_selector import (
    FrameSelector,
    FrameSelectorConfig,
    SelectionCriteria,
)
from personfromvid.data.detection_results import (
    FaceDetection,
    HeadPoseResult,
    PoseDetection,
    QualityMetrics,
)
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo


def make_sharp_image(height: int = 720, width: int = 1280, seed: int = 42) -> np.ndarray:
    """Noise image: high Laplacian variance (sharp)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_blurry_image(height: int = 720, width: int = 1280, value: int = 128) -> np.ndarray:
    """Solid-color image: zero Laplacian variance (blurry)."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_speckled_image(height: int = 720, width: int = 1280, seed: int = 7) -> np.ndarray:
    """Mostly solid image with a faint speckle: blurry but not zero variance."""
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, height, size=200)
    xs = rng.integers(0, width, size=200)
    image[ys, xs] = 255
    return image


def create_blur_test_frame(
    frame_id: str,
    image: np.ndarray,
    quality_score: float = 0.8,
    pose_confidence: float = 0.9,
    timestamp: float = 0.0,
) -> FrameData:
    """Create a test frame carrying a synthetic in-memory image."""
    height, width = image.shape[:2]
    frame = FrameData(
        frame_id=frame_id,
        file_path=Path(f"/test/{frame_id}.jpg"),
        source_info=SourceInfo(
            video_timestamp=timestamp,
            extraction_method="test",
            original_frame_number=0,
            video_fps=30.0,
        ),
        image_properties=ImageProperties(
            width=width,
            height=height,
            channels=3,
            file_size_bytes=100000,
            format="JPG",
        ),
        quality_metrics=QualityMetrics(
            laplacian_variance=100.0,
            sobel_variance=50.0,
            brightness_score=0.8,
            contrast_score=0.7,
            overall_quality=quality_score,
            quality_issues=[],
            usable=True,
        ),
        face_detections=[
            FaceDetection(bbox=(150, 120, 250, 220), confidence=0.85, landmarks=None)
        ],
        pose_detections=[
            PoseDetection(
                bbox=(100, 100, 300, 400),
                confidence=pose_confidence,
                keypoints={},
                pose_classifications=[("standing", 0.9)],
            )
        ],
        head_poses=[
            HeadPoseResult(
                yaw=0.0, pitch=0.0, roll=0.0,
                confidence=0.7, direction="front", face_id=0,
            )
        ],
    )
    # Inject the synthetic image directly; never touches disk
    frame._image = image
    return frame


def make_criteria(max_frames: int = 5) -> SelectionCriteria:
    return SelectionCriteria(
        min_frames_per_category=1,
        max_frames_per_category=max_frames,
        min_quality_threshold=0.5,
        face_size_weight=0.3,
        quality_weight=0.6,
        diversity_threshold=0.8,
        temporal_diversity_threshold=2.0,
    )


def selected_pose_ids(summary, category: str = "standing"):
    return [f.frame_id for f in summary.pose_selections[category].selected_frames]


class TestDefaultDisabledZeroOverhead:
    """blur_threshold=0 (the default) must not compute a single blur score
    and must reproduce the legacy selection frame-for-frame."""

    def test_default_config_computes_no_blur_scores(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("blur score computed while filter disabled")

        monkeypatch.setattr(FrameSelector, "_get_blur_score", boom)

        frames = [
            create_blur_test_frame(
                f"frame{i}", make_sharp_image(seed=i), timestamp=i * 5.0
            )
            for i in range(4)
        ]
        selector = FrameSelector(make_criteria())  # default config: disabled
        summary = selector.select_best_frames(frames)

        assert summary.total_selected > 0
        for frame in frames:
            assert not any(
                key.startswith("blur_score") for key in frame.debug_info
            ), "blur cache entries written while filter disabled"

    def test_disabled_matches_legacy_selection_exactly(self):
        def build_frames():
            return [
                create_blur_test_frame(
                    f"frame{i}",
                    make_sharp_image(seed=i) if i % 2 == 0 else make_blurry_image(),
                    quality_score=0.9 - i * 0.05,
                    timestamp=i * 5.0,
                )
                for i in range(6)
            ]

        legacy_frames = build_frames()
        legacy_summary = FrameSelector(make_criteria()).select_best_frames(legacy_frames)

        disabled_frames = build_frames()
        disabled_summary = FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=0.0)
        ).select_best_frames(disabled_frames)

        assert selected_pose_ids(disabled_summary) == selected_pose_ids(legacy_summary)
        for legacy, disabled in zip(legacy_frames, disabled_frames):
            assert legacy.selections.final_output == disabled.selections.final_output
            assert (
                legacy.selections.category_scores
                == disabled.selections.category_scores
            )


class TestBlurScoring:
    def setup_method(self):
        self.selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))

    def test_sharp_scores_above_blurry(self):
        sharp = create_blur_test_frame("sharp", make_sharp_image())
        blurry = create_blur_test_frame("blurry", make_blurry_image())

        sharp_score = self.selector._get_blur_score(sharp)
        blurry_score = self.selector._get_blur_score(blurry)

        assert sharp_score > blurry_score
        assert blurry_score == pytest.approx(0.0)

    def test_relative_order_preserved_across_resolutions(self):
        """Same content at 4K and 720p: sharp stays above blurry at both."""
        sharp_4k = create_blur_test_frame("sharp_4k", make_sharp_image(2160, 3840, seed=1))
        sharp_720 = create_blur_test_frame("sharp_720", make_sharp_image(720, 1280, seed=1))
        blurry_4k = create_blur_test_frame("blurry_4k", make_blurry_image(2160, 3840))
        blurry_720 = create_blur_test_frame("blurry_720", make_blurry_image(720, 1280))

        s4k = self.selector._get_blur_score(sharp_4k)
        s720 = self.selector._get_blur_score(sharp_720)
        b4k = self.selector._get_blur_score(blurry_4k)
        b720 = self.selector._get_blur_score(blurry_720)

        assert s4k > b4k
        assert s720 > b720
        # One shared threshold separates sharp from blurry at both resolutions
        threshold = max(b4k, b720)
        assert s4k > threshold and s720 > threshold

    def test_original_image_not_modified(self):
        image = make_sharp_image()
        snapshot = image.copy()
        frame = create_blur_test_frame("frame", image)

        self.selector._get_blur_score(frame)

        np.testing.assert_array_equal(frame.image, snapshot)

    def test_score_cached_in_frame_metadata(self):
        frame = create_blur_test_frame("frame", make_sharp_image())
        score = self.selector._get_blur_score(frame)

        cache_key = self.selector._blur_cache_key()
        assert frame.debug_info[cache_key] == score

    def test_cache_key_embeds_normalization_size(self):
        frame = create_blur_test_frame("frame", make_sharp_image())

        selector_512 = FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=1.0, blur_resize_short_edge=512)
        )
        selector_256 = FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=1.0, blur_resize_short_edge=256)
        )

        selector_512._get_blur_score(frame)
        assert "blur_score_s512" in frame.debug_info
        assert "blur_score_s256" not in frame.debug_info

        # Different normalization size must not read the stale 512 entry
        selector_256._get_blur_score(frame)
        assert "blur_score_s256" in frame.debug_info

    def test_scores_reused_when_threshold_changes(self, monkeypatch):
        """Re-running with a different threshold must not recompute scores."""
        frames = [
            create_blur_test_frame(f"frame{i}", make_sharp_image(seed=i), timestamp=i * 5.0)
            for i in range(3)
        ]

        laplacian_calls = 0
        real_laplacian = cv2.Laplacian

        def counting_laplacian(*args, **kwargs):
            nonlocal laplacian_calls
            laplacian_calls += 1
            return real_laplacian(*args, **kwargs)

        monkeypatch.setattr(frame_selector_module.cv2, "Laplacian", counting_laplacian)

        FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=1.0)
        ).select_best_frames(frames)
        calls_after_first_run = laplacian_calls
        assert calls_after_first_run > 0

        # Same frames, new selector, different threshold: cache must serve
        FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=1e9)
        ).select_best_frames(frames)
        assert laplacian_calls == calls_after_first_run


class TestBlurThresholdFiltering:
    def test_blurry_frames_removed_from_window(self):
        sharp_frames = [
            create_blur_test_frame(f"sharp{i}", make_sharp_image(seed=i), timestamp=i * 5.0)
            for i in range(2)
        ]
        blurry_frames = [
            create_blur_test_frame(
                f"blurry{i}", make_blurry_image(),
                quality_score=0.95,  # higher composite score, still must be filtered
                timestamp=(i + 2) * 5.0,
            )
            for i in range(2)
        ]
        frames = sharp_frames + blurry_frames

        probe = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        sharp_score = probe._get_blur_score(sharp_frames[0])
        threshold = sharp_score / 2  # between sharp and solid-blurry scores

        selector = FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=threshold)
        )
        summary = selector.select_best_frames(frames)

        assert set(selected_pose_ids(summary)) == {"sharp0", "sharp1"}

    def test_threshold_boundary_is_inclusive(self):
        """A frame scoring exactly at the threshold is kept (>= comparison)."""
        sharp = create_blur_test_frame("sharp", make_sharp_image(), timestamp=0.0)
        blurry = create_blur_test_frame("blurry", make_blurry_image(), timestamp=5.0)

        probe = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        exact_threshold = probe._get_blur_score(sharp)

        selector = FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=exact_threshold)
        )
        summary = selector.select_best_frames([sharp, blurry])

        assert selected_pose_ids(summary) == ["sharp"]

    def test_filtering_preserves_score_order(self):
        frames = [
            create_blur_test_frame(
                f"sharp{i}", make_sharp_image(seed=i),
                quality_score=0.9 - i * 0.1, timestamp=i * 5.0,
            )
            for i in range(3)
        ]
        selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        scored = [(f, 0.9 - i * 0.1) for i, f in enumerate(frames)]

        filtered = selector._apply_blur_filter(scored)

        assert [f.frame_id for f, _ in filtered] == ["sharp0", "sharp1", "sharp2"]


class TestAllBlurryFallback:
    def test_window_never_left_empty(self):
        frames = [
            create_blur_test_frame(f"blur{i}", make_blurry_image(), timestamp=i * 5.0)
            for i in range(3)
        ]
        selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        summary = selector.select_best_frames(frames)

        assert len(summary.pose_selections["standing"].selected_frames) == 1

    def test_fallback_picks_highest_blur_score_within_top_fraction(self):
        """Top 30% pool (ceil(4*0.3)=2 frames); the sharpest in-pool frame wins
        even when its composite score is lower."""
        frames = [
            create_blur_test_frame("best_composite", make_blurry_image(),
                                   quality_score=0.95, timestamp=0.0),
            create_blur_test_frame("sharper", make_speckled_image(),
                                   quality_score=0.85, timestamp=5.0),
            create_blur_test_frame("third", make_blurry_image(),
                                   quality_score=0.75, timestamp=10.0),
            create_blur_test_frame("fourth", make_blurry_image(),
                                   quality_score=0.65, timestamp=15.0),
        ]
        selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1e9))
        summary = selector.select_best_frames(frames)

        assert selected_pose_ids(summary) == ["sharper"]

    def test_fallback_tie_breaks_by_composite_score_then_timestamp(self):
        """Identical blur scores: higher composite score wins; identical
        composite scores: earlier timestamp wins."""
        # Equal blur, different composite scores -> highest composite wins
        frames = [
            create_blur_test_frame("low_comp", make_blurry_image(),
                                   quality_score=0.7, timestamp=0.0),
            create_blur_test_frame("high_comp", make_blurry_image(),
                                   quality_score=0.9, timestamp=5.0),
        ]
        selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        summary = selector.select_best_frames(frames)
        assert selected_pose_ids(summary) == ["high_comp"]

        # Equal blur and composite scores -> earliest timestamp wins
        # (4 frames so the top-30% pool, ceil(4*0.3)=2, contains both tied frames)
        frames = [
            create_blur_test_frame("later", make_blurry_image(),
                                   quality_score=0.9, pose_confidence=0.9, timestamp=10.0),
            create_blur_test_frame("earlier", make_blurry_image(),
                                   quality_score=0.9, pose_confidence=0.9, timestamp=2.0),
            create_blur_test_frame("filler1", make_blurry_image(),
                                   quality_score=0.6, timestamp=20.0),
            create_blur_test_frame("filler2", make_blurry_image(),
                                   quality_score=0.5, timestamp=30.0),
        ]
        selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        summary = selector.select_best_frames(frames)
        assert selected_pose_ids(summary) == ["earlier"]

    def test_fallback_small_window_keeps_at_least_one(self):
        """A single-frame all-blurry window still yields that one frame."""
        frames = [create_blur_test_frame("only", make_blurry_image(), timestamp=0.0)]
        selector = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        summary = selector.select_best_frames(frames)

        assert selected_pose_ids(summary) == ["only"]

    def test_fallback_pool_rounds_up(self):
        """ceil rounding: 4 frames -> pool of 2, so a sharper frame ranked
        second by composite score is eligible for the fallback."""
        frames = [
            create_blur_test_frame("first", make_blurry_image(),
                                   quality_score=0.95, timestamp=0.0),
            create_blur_test_frame("second_sharper", make_speckled_image(),
                                   quality_score=0.85, timestamp=5.0),
            create_blur_test_frame("third_sharpest", make_sharp_image(),
                                   quality_score=0.75, timestamp=10.0),
            create_blur_test_frame("fourth", make_blurry_image(),
                                   quality_score=0.65, timestamp=15.0),
        ]
        # Threshold above even the sharp frame -> whole window counts as blurry
        probe = FrameSelector(make_criteria(), FrameSelectorConfig(blur_threshold=1.0))
        top_score = probe._get_blur_score(frames[2])

        selector = FrameSelector(
            make_criteria(), FrameSelectorConfig(blur_threshold=top_score * 2)
        )
        summary = selector.select_best_frames(frames)

        # "third_sharpest" is sharpest overall but outside the top-30% pool
        # (pool = ceil(4*0.3) = 2 frames), so the in-pool speckled frame wins
        assert selected_pose_ids(summary) == ["second_sharper"]


class TestFrameSelectorConfigValidation:
    def test_defaults_disabled(self):
        config = FrameSelectorConfig()
        assert config.blur_threshold == 0.0
        assert not config.blur_filter_enabled

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError):
            FrameSelectorConfig(blur_threshold=-1.0)

    def test_invalid_resize_edge_rejected(self):
        with pytest.raises(ValueError):
            FrameSelectorConfig(blur_resize_short_edge=0)

    def test_invalid_fallback_ratio_rejected(self):
        with pytest.raises(ValueError):
            FrameSelectorConfig(blur_fallback_ratio=0.0)
        with pytest.raises(ValueError):
            FrameSelectorConfig(blur_fallback_ratio=1.5)
