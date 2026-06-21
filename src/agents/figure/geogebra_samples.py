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
    # ===================== 基础图型（PRD-A-021 R3b 补，治「数轴乱画/图型有限」）=====================
    {
        # 🔴 数轴专项模板（治「数轴乱画」根因：opus 之前无据可仿，自由发挥画歪）。
        # 数轴 = 一条**水平线段** + 正方向**箭头**（Vector，绝不用 Arrow）+ 原点 0 + 若干刻度点 + 点上方
        # Text 标数值。轴本身用一条 Segment/Line，刻度用小竖线 Segment 或直接点，标注用 Text。
        "kind": "数轴(number_line)",
        "commands": [
            "axis=Segment((-4,0),(4,0))",
            "arrow=Vector((4,0),(4.6,0))",
            "O=(0,0)", "P=(2,0)", "Q=(-3,0)",
            't0=Text("0",(0,-0.45))', 't1=Text("2",(2,-0.45))', 't2=Text("-3",(-3,-0.45))',
        ],
        "hide": ["O"],
        "note": ("数轴 = 水平 Segment 作轴线 + **Vector(末端,末端外一点) 作正方向箭头**"
                 "（🔴 绝不写 Arrow——GeoGebra 无此命令、必失败整图作废）；原点/要标的点用自由点 "
                 "A=(x,0)，**数值标注一律用 Text(\"数\",(x,-0.45)) 放点下方**（别用 Angle/别在轴上乱画）。"
                 "实心标记点保留可见（point_size 默认即可），纯刻度辅助点放 hide。负数直接写 Text(\"-3\",…)。"
                 "区间/不等式解集：端点空心(开)用 Text(\"○\",…)、实心(闭)用大点，区间段用粗 Segment 叠在轴上。"),
    },
    {
        "kind": "平面直角坐标系(cartesian)",
        "commands": [
            "A=(2,3)", "B=(-1,-2)", "O=(0,0)",
            'la=Text("A(2,3)",(2.2,3.2))', 'lb=Text("B(-1,-2)",(-0.8,-2.4))',
        ],
        "hide": ["O"],
        "axes": True,
        "note": ("坐标系题**置顶层 axes=true**（render 自动画 x/y 轴 + 原点 + 网格刻度，不用手画轴）；"
                 "只需放要标的点 A=(x,y) + Text 标坐标。象限/网格交给 axes，别自己用 Segment 拼轴。"),
    },
    {
        "kind": "抛物线/二次函数图象(parabola)",
        "commands": [
            "f(x)=x^2-2x-3",
            "V=(1,-4)", "A=(-1,0)", "B=(3,0)", "C=(0,-3)",
            'lv=Text("顶点",(1.2,-4.3))',
        ],
        "axes": True,
        "note": ("二次函数图象：直接 f(x)=a x^2+b x+c（GeoGebra 自动绘曲线），axes=true 出坐标系；"
                 "顶点/与轴交点用自由点标出（坐标可由你算或让引擎 Intersect(f,xAxis)）。"
                 "一次函数同理 g(x)=k x+b；反比例 h(x)=k/x。曲线本身别用 Polygon/Segment 拼。"),
    },
    {
        "kind": "三角形(triangle)",
        "commands": [
            "A=(0,0)", "B=(6,0)", "C=(2,4)", "tri=Polygon(A,B,C)",
        ],
        "note": ("三角形 = 三顶点自由点 + Polygon(A,B,C)。要画高/中线/角平分线用 "
                 "PerpendicularLine/Segment(顶点,Midpoint(..))/AngleBisector，构造的垂足/中点等中间点放 hide。"
                 "直角三角形在直角顶点用构型记号（题面要求时），等腰/等边两腰加等长刻度记号。"),
    },
    {
        "kind": "圆(circle)",
        "commands": [
            "O=(0,0)", "c=Circle(O,3)", "A=(3,0)", "B=(0,3)",
            "chord=Segment(A,B)",
        ],
        "note": ("圆 = Circle(圆心,半径) 或 Circle(A,B,C) 过三点；圆心 Center(c)。"
                 "弦/切线/半径用 Segment；圆周上的点用 Point(c) 或自由点放到圆上。"
                 "🔴 注意区分：圆 = circle（平面），圆柱/圆锥/球 = solid（立体，走斜二测投影），别混。"),
    },
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
        if s.get("axes"):
            lines.append("  axes: true（坐标系题置顶层 axes=true，render 自动画轴+网格，不手画轴）")
        if s.get("relabel"):
            lines.append("  relabel: " + ", ".join(f"{k}→{v}" for k, v in s["relabel"].items()))
        if s.get("angle_labels"):
            lines.append("  angle_labels: " + ", ".join(f"{k}→{v}" for k, v in s["angle_labels"].items()))
    return "\n".join(lines)
