# -*- coding: utf-8 -*-
"""PRD-C-100 B5 单一全局日预算护栏（G7）：累计/超额判定 + 拦截/降级（非静默）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import cost_guard  # noqa: E402
from core.settings import settings  # noqa: E402


class TestBudgetStatus:
    def test_limit_off_never_exceeds(self, monkeypatch):
        monkeypatch.setattr(settings, "GLOBAL_DAILY_BUDGET_YUAN", None)
        cost_guard._cache["ts"] = 0.0
        st = cost_guard.budget_status()
        assert st["exceeded"] is False and st["limit"] is None
        assert cost_guard.is_budget_exceeded() is False

    def test_zero_limit_off(self, monkeypatch):
        monkeypatch.setattr(settings, "GLOBAL_DAILY_BUDGET_YUAN", 0)
        assert cost_guard.is_budget_exceeded() is False

    def test_high_limit_not_exceeded(self, monkeypatch):
        # 极高阈值 → 不超（无论今日实际花费）
        monkeypatch.setattr(settings, "GLOBAL_DAILY_BUDGET_YUAN", 1_000_000.0)
        cost_guard._cache["ts"] = 0.0
        st = cost_guard.budget_status()
        assert st["exceeded"] is False
        assert st["remaining"] is not None and st["remaining"] > 0

    def test_tiny_limit_exceeded_when_spend_exists(self, monkeypatch):
        # 极低阈值 0.000001 → 只要今日有任何花费就超（真 conv_trace 今日 e2e 已产生花费）
        monkeypatch.setattr(settings, "GLOBAL_DAILY_BUDGET_YUAN", 0.000001)
        cost_guard._cache["ts"] = 0.0
        spend = cost_guard.today_spend_yuan(force=True)
        # 今日有 e2e 花费 → 超；若环境无花费(全新库)则 spend=0 不超（断言随实际）
        assert cost_guard.is_budget_exceeded() == (spend >= 0.000001)

    def test_spend_is_nonnegative_float(self):
        s = cost_guard.today_spend_yuan(force=True)
        assert isinstance(s, float) and s >= 0.0

    def test_cache_avoids_requery(self, monkeypatch):
        # 第二次调用（缓存内）不重查：先 force 填缓存，再普通调用返同值
        monkeypatch.setattr(settings, "GLOBAL_DAILY_BUDGET_YUAN", 999.0)
        v1 = cost_guard.today_spend_yuan(force=True)
        v2 = cost_guard.today_spend_yuan()  # 命中缓存
        assert v1 == v2
