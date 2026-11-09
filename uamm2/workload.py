"""Synthetic attention workload (NOT a real LLM).

Produces per-step (q, k, v) with structure that exercises the system:
  * a few attention-sink tokens and sporadic heavy-hitter tokens,
  * outlier key channels (the per-channel-key-quantization motivation),
  * topic phases whose blocks become hot again when the query returns to an old topic
    (so cold blocks must be promoted back),
  * local recency in the query.
Replace with real model activations for any real conclusion.
"""
import numpy as np


class SyntheticWorkload:
    def __init__(self, d: int, seed: int = 7, n_topics: int = 4, phase_len: int = 96,
                 sharpness: float = 4.0):
        self.d = d
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.channel_scale = np.ones(d)
        self.channel_scale[r.choice(d, size=max(1, d // 16), replace=False)] = 6.0   # key outliers
        self.topics = r.normal(size=(n_topics, d))
        self.topics /= np.linalg.norm(self.topics, axis=1, keepdims=True)
        self.heavy_dir = r.normal(size=d)
        self.heavy_dir /= np.linalg.norm(self.heavy_dir)
        self.phase_len = phase_len
        self.sharpness = sharpness   # scales query norm -> softmax peakedness
        self.n_topics = n_topics
        self._keys = []

    def topic_of(self, t: int) -> int:
        return (t // self.phase_len) % self.n_topics

    def query_topic(self, t: int) -> int:
        phase = t // self.phase_len
        # revisit older topics periodically so old blocks turn hot again
        return (phase if phase % 3 else max(0, phase - 2)) % self.n_topics

    def step(self, t: int):
        r, d = self.rng, self.d
        heavy = t < 4 or r.random() < 0.03
        k = r.normal(size=d) * 0.5 * self.channel_scale
        k = k + 3.0 * np.sqrt(d) / 8 * self.topics[self.topic_of(t)]
        if heavy:
            k = k + 4.0 * np.sqrt(d) / 8 * self.heavy_dir
        v = r.normal(size=d)
        q = 0.3 * r.normal(size=d) + 2.0 * np.sqrt(d) / 8 * self.topics[self.query_topic(t)]
        q = q + 1.0 * np.sqrt(d) / 8 * self.heavy_dir
        if self._keys:
            j = int(r.integers(1, min(len(self._keys), 12) + 1))
            q = q + 0.5 * self._keys[-j] / np.linalg.norm(self._keys[-j]) * np.sqrt(d) / 8
        self._keys.append(k)
        q = q * self.sharpness
        return q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
