# -*- coding: utf-8 -*-
"""审计冒烟：故意发不带 subjectId / difficult / dim4Difficulty 的母题入库 BO，
验证 BE create() 双兜底(subject_id→"0"、difficult→2)生效、不再 500。
修前=insert NOT NULL 违约 500；修后=code:1 且 difficult=2 subjectId=0。"""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agents.variant_support import RuoyiClient


async def main():
    c = RuoyiClient()
    await c.login()
    body = {
        "questionType": 1,
        "stem": "审计冒烟·subjectId/difficult 双空 $a=\\sqrt{2}$，则 $a$ 满足（）",
        "answer": "无",
        "labelStatus": 1,
        "labeledBy": "audit-smoke-761546c",
        # 🔴 故意不传 subjectId / difficult / dim4Difficulty → 触发 BE 双 NOT NULL 兜底
    }
    try:
        res = await c.create_question(body)
        d = res.get("difficult") if isinstance(res, dict) else None
        sid = str(res.get("subjectId")) if isinstance(res, dict) else None
        qid = res.get("id") if isinstance(res, dict) else None
        print(f"CREATE OK (envelope code==1) id={qid}")
        print(f"  difficult={d}  subjectId={sid}")
        ok = (d == 2 and sid == "0")
        print(f"SMOKE {'PASS' if ok else 'CHECK'} — 双兜底{'生效' if ok else '待核(值非2/0)'}")
    except Exception as e:
        print(f"SMOKE FAIL — create 抛错(修复未生效?): {type(e).__name__}: {repr(e)[:200]}")
    finally:
        await c.aclose()


asyncio.run(main())
