# -*- coding: utf-8 -*-
"""6 图型 canonical GeoGebra 命令样例（B-mathfig render 抽验已 0-fail 验过 2026-06-17）。

喂 B3 opus「翻 GeoGebra 命令」prompt：旋转/平移/对称/折叠/伪3D 立体(斜二测)/三视图(2D拼框)。
recon 结论：这些全 GeoGebra 主路径零引擎改造（旧 matplotlib REJECT 条文不适用）。
🔴 变换题的「像」放进 dashed；隐藏辅助构造线放 hide；立体隐藏棱用 dashed。
🔴 命令语义坑（喂进 prompt 防返工，源 = mathfig mcp_server docstring）：
   - 自由点直接 A=(2,3)，**别用 Point((2,3))**（会失败）；
   - 派生点用 Intersect/Midpoint/Rotate/Reflect/Translate 让引擎算，别手填坐标；
   - 别在 commands 里写 SetLineStyle（会把对象渲染没）—— 虚线走 dashed 参数。
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
        "note": "Rotate(对象,角度°,中心)；像的边用 dashed 区分原图。",
    },
    {
        "kind": "平移",
        "commands": [
            "A=(0,0)", "B=(4,0)", "C=(1,3)", "t=Polygon(A,B,C)", "u=Vector((5,1))",
            "A2=Translate(A,u)", "B2=Translate(B,u)", "C2=Translate(C,u)", "t2=Polygon(A2,B2,C2)",
        ],
        "dashed": ["t2"],
        "note": "Translate(对象,向量)；先 Vector((dx,dy)) 定平移向量。",
    },
    {
        "kind": "对称",
        "commands": [
            "A=(1,1)", "B=(4,2)", "C=(2,4)", "t=Polygon(A,B,C)", "ax=Line((0,0),(0,1))",
            "A2=Reflect(A,ax)", "B2=Reflect(B,ax)", "C2=Reflect(C,ax)", "t2=Polygon(A2,B2,C2)",
        ],
        "dashed": ["t2"],
        "hide": ["ax"],
        "note": "Reflect(对象,对称轴线/点)；对称轴用 hide 隐去，像用 dashed。",
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
    return "\n".join(lines)
