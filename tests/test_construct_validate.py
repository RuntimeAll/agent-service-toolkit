# -*- coding: utf-8 -*-
"""构件 DSL 浅校验（dsl_schema.validate_construct）+ 双格式路由（validate_dsl）测试。

构件层 = book-ui/public/geo-engine/figure-builder.js（几何深校验在前端，75 项 node 不变量测保证）；
toolkit 侧只做结构浅校验。本测试守：① 好 build 过 ② 坏 build 挡 ③ 低层 DSL 不回归。
"""
from agents.figure import dsl_schema as S


def test_good_construct_passes():
    spec = {"build": [
        {"id": "T", "shape": "triangle", "kind": "right", "rightAt": "C", "legs": [6, 8]},
        {"add": "median", "of": "T", "from": "C", "label": "D"},
        {"mark": "rightangle", "of": "T", "at": "C"},
    ]}
    ok, errs = S.validate_dsl(spec)
    assert ok, errs


def test_unknown_shape_rejected():
    ok, errs = S.validate_dsl({"build": [{"shape": "hyperboloid"}]})
    assert not ok and any("白名单" in e for e in errs)


def test_add_without_of_rejected():
    ok, errs = S.validate_dsl({"build": [{"add": "median", "from": "A"}]})
    assert not ok and any("of" in e for e in errs)


def test_two_kind_keys_rejected():
    ok, errs = S.validate_dsl({"build": [{"shape": "triangle", "add": "median"}]})
    assert not ok


def test_empty_build_rejected():
    ok, errs = S.validate_dsl({"build": []})
    assert not ok


def test_transform_and_all_shapes_whitelist():
    spec = {"build": [
        {"id": "T", "shape": "triangle", "kind": "sss", "sides": [3, 4, 5]},
        {"transform": "reflect", "of": "T", "axis": "y"},
    ]}
    ok, errs = S.validate_dsl(spec)
    assert ok, errs


def test_low_level_dsl_not_regressed():
    # 低层 DSL 仍走深校验（type 白名单 + 必填 + 引用）——构件路由不能影响它。
    good = {"objects": [
        {"id": "A", "type": "point", "coords": [0, 0], "label": "A"},
        {"id": "B", "type": "point", "coords": [3, 0]},
        {"type": "segment", "points": ["A", "B"]},
    ]}
    assert S.validate_dsl(good)[0]
    bad = {"objects": [{"id": "A", "type": "wat"}]}
    assert not S.validate_dsl(bad)[0]
    # 引用未定义 id 仍被挡
    dangling = {"objects": [{"type": "segment", "points": ["X", "Y"]}]}
    assert not S.validate_dsl(dangling)[0]
