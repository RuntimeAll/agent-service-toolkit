# -*- coding: utf-8 -*-
"""PRD-C-100 B1b 计费 + 护栏配置化：opus cost 口径 + 全节点 max_tokens 上限断言。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import relay_pool  # noqa: E402
from core.settings import settings  # noqa: E402


class TestOpusCostKoujing:
    """G6：cost = prompt×in价 + completion×out价（口径对，非核某数值）。"""

    def test_opus_has_builtin_default_price(self):
        # 内建默认价兜底 → opus cost 永不为 None（即便 .env 漏配）
        relay_pool._prices_cache = None  # 清缓存
        c = relay_pool.cost_yuan("claude-opus-4-8", 1000, 500)
        assert c is not None
        # 默认价 in=0.007/1k out=0.035/1k → 1000*0.007/1000 + 500*0.035/1000 = 0.007+0.0175
        assert abs(c - 0.0245) < 1e-9

    def test_unknown_model_cost_none(self):
        relay_pool._prices_cache = None
        assert relay_pool.cost_yuan("no-such-model", 1000, 500) is None

    def test_missing_tokens_cost_none(self):
        relay_pool._prices_cache = None
        assert relay_pool.cost_yuan("claude-opus-4-8", None, 500) is None
        assert relay_pool.cost_yuan("claude-opus-4-8", 1000, None) is None

    def test_env_overrides_default(self, monkeypatch):
        # RELAY_PRICES(.env) 覆盖内建默认（D5 配置化）
        relay_pool._prices_cache = None
        monkeypatch.setattr(settings, "RELAY_PRICES",
                            '{"claude-opus-4-8":{"in":0.01,"out":0.05}}')
        c = relay_pool.cost_yuan("claude-opus-4-8", 1000, 1000)
        relay_pool._prices_cache = None  # 复原缓存防污染其他测
        assert abs(c - (0.01 + 0.05)) < 1e-9


class TestMaxTokensGuardrail:
    """G8：母题节点宽护栏 + 全节点 max_tokens 有上限（无失控/无截断必要输出）。"""

    def test_mother_guardrail_wider_than_default(self):
        # 母题一把节点护栏 12288 > 默认 4096（B0 H5：峰值 5158，4096 会截断）
        assert settings.MOTHER_OPUS_MAX_TOKENS > settings.VARIANT_MAX_TOKENS
        assert settings.MOTHER_OPUS_MAX_TOKENS >= 8192

    def test_all_node_caps_positive(self):
        # 全节点 max_tokens 上限均 > 0（无无界调用）
        assert settings.VARIANT_MAX_TOKENS > 0
        assert settings.MOTHER_OPUS_MAX_TOKENS > 0
        assert settings.VARIANT_REGEN_MAX_TOKENS > 0

    def test_mother_guardrail_above_b0_observed_peak(self):
        # B0 H5 实测母题 completion 峰值 5158（零截断）→ 护栏须留头
        B0_OBSERVED_PEAK = 5158
        assert settings.MOTHER_OPUS_MAX_TOKENS > B0_OBSERVED_PEAK
