from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.backend.transcript_cache import TranscriptPrefixCacheManager
from tests.helpers import TinyModelWithCache


class TrimmableModelWithCache(TinyModelWithCache):
    def make_cache(self):
        return [KVCache() for _ in self.layers]


class UntrimmableModelWithCache(TinyModelWithCache):
    def make_cache(self):
        return [ArraysCache(size=2)]


def _manager(tmp_path: Path) -> TranscriptPrefixCacheManager:
    return TranscriptPrefixCacheManager(tmp_path / "prefix_cache", max_bytes=1 << 20)


def test_transcript_cache_untrimmable_safety(tmp_path: Path):
    """When a model's cache is not trimmable, exact_hash and longer_hash must not claim hit.

    Otherwise, uncached_tail=[tokens[-1]] would evaluate the prompt tail token again,
    causing prompt echo and duplicate tokens.
    """
    manager = _manager(tmp_path)

    # Create an untrimmable cache entry (using ArraysCache)
    arrays_cache = [ArraysCache(size=2)]
    arrays_cache[0].state = [mx.ones((1, 4, 8)), mx.ones((1, 8, 8, 8))]
    manager.enqueue_persist("ns", [10, 20, 30, 40], arrays_cache, "final")
    manager.wait_for_idle()

    # Exact lookup: since ArraysCache cannot trim 1 token, lookup should NOT claim exact hit
    # with uncached_tail=[40] because that would evaluate 40 twice!
    lookup_exact = manager.lookup("ns", [10, 20, 30, 40])
    assert lookup_exact.prompt_cache is None
    assert lookup_exact.cached_tokens == 0
    assert lookup_exact.uncached_tail == [10, 20, 30, 40]

    # Longer lookup: prompt is [10, 20, 30], but cache has [10, 20, 30, 40].
    # Cannot trim down, so it should not return longer hit.
    lookup_longer = manager.lookup("ns", [10, 20, 30])
    assert lookup_longer.prompt_cache is None
    assert lookup_longer.cached_tokens == 0
    assert lookup_longer.uncached_tail == [10, 20, 30]

    manager.close()


def test_inference_backend_disables_transcript_cache_for_untrimmable_models(tmp_path: Path):
    """InferenceBackend._should_use_transcript_cache should return False for untrimmable models."""
    manager = _manager(tmp_path)
    backend = InferenceBackend(transcript_cache=manager)

    trimmable_model = TrimmableModelWithCache()
    untrimmable_model = UntrimmableModelWithCache()

    assert backend._should_use_transcript_cache(trimmable_model, "ns") is True
    assert backend._should_use_transcript_cache(untrimmable_model, "ns") is False

    manager.close()
