# -*- coding: utf-8 -*-
"""6 图型 canonical GeoGebra 命令样例（B-mathfig render 抽验已 0-fail 验过 2026-06-17）。

喂 B3 opus「翻 GeoGebra 命令」prompt：旋转/平移/对称/折叠/伪3D 立体(斜二测)/三视图(2D拼框)。
recon 结论：这些全 GeoGebra 主路径零引擎改造（旧 matplotlib REJECT 条文不适用）。
🔴 变换题的「像」放进 dashed；隐藏辅助构造线放 hide；立体隐藏棱用 dashed。
🔴 命令语义坑（喂进 prompt 防返工，源 = mathfig mcp_server docstring）：
   - 自由点直接 A=(2,3)，**别用 Point((2,3))**（会失败）；
   - 派生点用 Intersect/Midpoint/Rotate/Reflect/Translate 让引擎算，别手填坐标；
   - 别在 commands 里写 SetLineStyle（会把对象渲染没）—— 虚线走 dashed 参数。
🔴 文字/公式标注（2026-06-20 实测定，治「公式插不进」）：
   - 图里写公式/符号一律用 **Unicode**（√ ∠ ° ′ ² ³ ₁₂ ∥ ⊥ ≅ ∽ △ × ÷ ± π）放进 Text("…")；
     **无头渲染器不认 LaTeX 宏**（\\frac/\\sqrt 印反斜杠乱码、$…$ 也不行），分式写 a/b。
   - 旋转/对称/平移的「像」点（A′/B′）：标识符不能含撇号 → 命名 Ap/Bp，用顶层 **relabel**
     字段 {"Ap":"A′"} 让标签显示撇号，**别手放 Text("A'",…)**（与自动标签双标打架、坐标必偏）。
"""

# 每条 = {图型, commands, dashed?, hide?, note}（note 给 opus 看构造范式）
SAMPLES: list[dict] = [
    {
        "kind": "旋转",
        "commands": [
            "A=(0,0)", "B=(6,0)", "C=(3.2*cos(75°),3.2*sin(75°))", "tri=Polygon(A,B,C)",
            "Bp=Rotate(B,30°,A)", "Cp=Rotate(C,30°,A)", "tri2=Polygon(A,Bp,Cp)",
        ],
        "dashed": ["tri2"],
        "relabel": {"Bp": "B′", "Cp": "C′"},
        "note": ("Rotate(对象,角度°,中心)；像的边用 dashed 区分原图。"
                 "🔴 像点命名 Bp/Cp（标识符不能含撇号），用 relabel 让图上显示 B′/C′——"
                 "**别手放 Text(\"B'\",…) 撇号标签**（会和自动标签双标打架且坐标必偏）。"),
    },
    {
        "kind": "平移",
        "commands": [
            "A=(0,0)", "B=(4,0)", "C=(1,3)", "t=Polygon(A,B,C)", "u=Vector((5,1))",
            "Ap=Translate(A,u)", "Bp=Translate(B,u)", "Cp=Translate(C,u)", "t2=Polygon(Ap,Bp,Cp)",
        ],
        "dashed": ["t2"],
        "relabel": {"Ap": "A′", "Bp": "B′", "Cp": "C′"},
        "note": ("Translate(对象,向量)；先 Vector((dx,dy)) 定平移向量。"
                 "像点 Ap/Bp/Cp 用 relabel 映射 A′/B′/C′。"),
    },
    {
        "kind": "对称",
        "commands": [
            "A=(1,1)", "B=(4,2)", "C=(2,4)", "t=Polygon(A,B,C)", "ax=Line((0,0),(0,1))",
            "Ap=Reflect(A,ax)", "Bp=Reflect(B,ax)", "Cp=Reflect(C,ax)", "t2=Polygon(Ap,Bp,Cp)",
        ],
        "dashed": ["t2"],
        "hide": ["ax"],
        "relabel": {"Ap": "A′", "Bp": "B′", "Cp": "C′"},
        "note": ("Reflect(对象,对称轴线/点)；对称轴用 hide 隐去，像用 dashed。"
                 "像点 Ap/Bp/Cp 用 relabel 映射 A′/B′/C′。"),
    },
    {
        "kind": "标注角(题面要求标度数/∠1∠2/示意符号)",
        "commands": [
            "A=(0,0)", "B=(6,0)", "C=(2,4)", "tri=Polygon(A,B,C)",
            "a1=Angle(B,A,C)", "a2=Angle(C,B,A)",
            'l3=Text("AB=6",(2.6,-0.5))',
        ],
        "vals": ["a1", "a2"],
        "angle_labels": {"a2": "β"},
        "note": ("题面明确要标的角（∠1/∠2/∠BAC…）**必须**用 Angle(P,V,Q) 画出角记号——"
                 "V 为顶点放中间：∠BAC=Angle(B,A,C)、∠ABC=Angle(C,B,A)；"
                 "点序须让有向角扫角 ≤180°（否则渲成优角），扫超平角就把首尾两点调换。"
                 "🔴🔴 要标角的**度数**：只画 Angle()——引擎自动按角平分线标实测度数，**绝不手放 "
                 "Text(\"20°\",(x,y))**（坐标靠猜必偏 + 双标打架）；同顶点多角弧引擎自动按角大小递增半径错开。"
                 "🔴 示意图（画的角≠要标的度数 / 标 α/β/∠1 等符号）→ 顶层 angle_labels 覆盖该角文字"
                 "（此例 a2 标「β」而非实测度数）；要标实测度数的角（a1）只画 Angle 不进 angle_labels。"
                 "🔴 边长/AB=6/非角度的自由标注仍用 Text（Unicode：∠ ° ² √ ∥ ⊥ ′ ₁₂，绝不写 LaTeX 宏）。"
                 "🔴 列清单时凡题面写了「标出∠1、∠2」就各一条 Angle，绝不只画三角形不标角。"),
    },
    {
        "kind": "折叠",
        "commands": [
            "B=(0,0)", "C=(6,0)", "D=(6,6)", "A=(0,6)", "M=Midpoint(B,C)",
            "l=PerpendicularBisector(A,M)", "sCD=Segment(C,D)", "sAB=Segment(A,B)",
            "P=Intersect(l,sAB)", "Q=Intersect(l,sCD)", "crease=Segment(P,Q)",
            "D1=Reflect(D,l)", "i1=Segment(P,M)", "i2=Segment(M,D1)",
        ],
        "dashed": ["i1", "i2"],
        "hide": ["l"],
        "note": "折痕 = Segment；翻折像 = Reflect(点,折痕线)；翻折后的边用 dashed。",
    },
    {
        "kind": "伪3D立体(斜二测)",
        "commands": [
            "A=(0,0)", "B=(4,0)", "C=(5,1.5)", "D=(1,1.5)",
            "E=(0,3)", "F=(4,3)", "G=(5,4.5)", "H=(1,4.5)",
            "bottom=Polygon(A,B,C,D)", "top=Polygon(E,F,G,H)",
            "e1=Segment(A,E)", "e2=Segment(B,F)", "e3=Segment(C,G)", "e4=Segment(D,H)",
        ],
        "dashed": ["e4"],
        "note": "立体走 2D 斜二测投影：底/顶面 Polygon + 竖棱 Segment；被遮挡的棱用 dashed。覆盖教材绝大多数立体图，零引擎改造。",
    },
    {
        "kind": "三视图(2D拼框)",
        "commands": [
            "f1=Polygon((0,0),(2,0),(2,2),(0,2))", "f2=Polygon((3,0),(5,0),(5,2),(3,2))",
            "f3=Polygon((0,-3),(2,-3),(2,-1),(0,-1))",
            'tt1=Text("主视图",(0.3,2.3))', 'tt2=Text("左视图",(3.3,2.3))', 'tt3=Text("俯视图",(0.3,-0.7))',
        ],
        "note": "三视图 = 匿名 Segment/Polygon 拼正/侧/俯三框 + Text 标注；render 自动取景已支持全匿名构图。",
    },
]


def samples_prompt_block() -> str:
    """渲成喂 opus 翻命令 prompt 的样例块（含构造范式 note）。"""
    lines = ["================ GeoGebra 命令样例（按图型，照此范式构造） ================"]
    for s in SAMPLES:
        lines.append(f"【{s['kind']}】{s['note']}")
        lines.append("  commands: " + " ; ".join(s["commands"]))
        if s.get("dashed"):
            lines.append("  dashed: " + ", ".join(s["dashed"]))
        if s.get("hide"):
            lines.append("  hide: " + ", ".join(s["hide"]))
        if s.get("relabel"):
            lines.append("  relabel: " + ", ".join(f"{k}→{v}" for k, v in s["relabel"].items()))
        if s.get("angle_labels"):
            lines.append("  angle_labels: " + ", ".join(f"{k}→{v}" for k, v in s["angle_labels"].items()))
    return "\n".join(lines)
