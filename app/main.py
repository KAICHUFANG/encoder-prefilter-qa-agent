"""FastAPI 接口层。

    GET  /health     活着吗、模型在哪、几类标签
    GET  /labels     标签清单
    GET  /policies   意图 -> 处理策略（kb/skill/template/llm/human）
    POST /intent     一句话 -> 意图 + 置信度 + top_k
    POST /answer     一句话 -> 完整回答（先查库/技能，不行才升级）
    GET  /dashboard  门控实验面板
    POST /api/gate   面板里的在线试判

接口层只做「收参数、校验、转手」；推理在 app/model.py。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config
from app import executor, labels as label_mod
from app import routing
from app.model import predictor

app = FastAPI(title="客服意图识别 Agent", version="0.1.0")

# 商户库统一用 crosswoz 官方库：模型目录可能是 crosswoz_gate 等，没有自己的一份库
_OFFICIAL_KB = config.PROJ / "data" / "crosswoz" / "kb_official.json"
if not (config.DATA_DIR / "kb_official.json").exists() and _OFFICIAL_KB.exists():
    executor.KB_PATH = _OFFICIAL_KB
    executor.load_rows.cache_clear()


class IntentIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="用户的一句话")
    top_k: int = Field(3, ge=1, le=10, description="返回前 k 个候选")


class AnswerIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(3, ge=1, le=10)


def _detect_coverage(text: str) -> tuple[float | None, int]:
    from app import ood

    try:
        det = ood.detector()
        return det.coverage(text), det.max_unseen_run(text)
    except FileNotFoundError:
        return None, 0


@app.get("/health")
def health() -> dict[str, Any]:
    info: dict[str, Any] = {
        "ok": True,
        "model_dir": str(config.MODEL_OUT),
        "model_loaded": predictor.loaded,
        "model_load_seconds": getattr(predictor, "load_seconds", None),
        "labels": label_mod.count(),
        "config": (f"dataset={config.DATASET} model={config.MODEL_NAME} "
                   f"device={config.DEVICE} max_len={config.MAX_LEN} "
                   f"batch={config.BATCH_SIZE} epochs={config.EPOCHS} lr={config.LR}"),
    }
    return info


@app.get("/labels")
def labels() -> dict[str, Any]:
    return {"labels": label_mod.names(), "count": label_mod.count()}


@app.get("/policies")
def policies() -> dict[str, Any]:
    described = routing.describe()
    return {"dataset": config.DATASET, "table": described["table"],
            "actions": described["actions"],
            "review_floor": routing.REVIEW_FLOOR,
            "count": len(described["table"])}


@app.post("/answer")
def answer(body: AnswerIn) -> dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能是空白字符")

    try:
        cls = predictor.predict(text, top_k=body.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    coverage, unseen_run = _detect_coverage(text)
    decision = routing.decide(cls["intent"], cls["confidence"], coverage, unseen_run)
    out: dict[str, Any] = {"text": text, "intent": cls["intent"],
                           "confidence": cls["confidence"], "routing": decision}

    # 槽位是纯规则，成本为零：不管走哪条路都先抽出来
    slots: dict = {}
    if config.DATASET.startswith("enterprise"):
        from app.slots import extract as extract_slots

        slots = extract_slots(text)
        out["slots"] = slots

    if decision["action"] == "template":
        out.update({"mode": "template", "needs_llm": False,
                    "answer": "您好，我是旅游/制度问答助手，可以问我商户信息、营业时间、门票和制度条款。"})
        return out

    if decision["action"] == "skill":
        from app import skills

        result = skills.handle(cls["intent"], text)
        if result["ok"]:
            out.update({"mode": "skill", "answer": result["reply"],
                        "params": result["params"], "missing_slots": result["missing_slots"],
                        "source": result["source"], "needs_llm": False})
            return out
        out.update({"mode": "llm", "answer": None, "needs_llm": True,
                    "reason": result.get("reason", "技能无法处理")})
        return out

    if decision["action"] == "kb":
        if config.DATASET.startswith("enterprise"):
            from app import enterprise_answer

            res = enterprise_answer.answer(cls["intent"], slots, text)
            if res["ok"]:
                out.update({"mode": res["mode"], "answer": res["answer"],
                            "slots": slots, "computed": res.get("computed"),
                            "source": res["source"], "needs_llm": False})
                return out
            out.update({"mode": "kb_miss", "answer": None, "slots": slots,
                        "needs_llm": True, "reason": res["reason"]})
            return out

        try:
            result = executor.execute(cls["intent"], text, top_k=body.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        if result["ok"]:
            out.update({"mode": "kb", "answer": result["answer"],
                        "matched": result["matched"], "constraints": result["constraints"],
                        "source": result["source"], "needs_llm": False})
            return out
        out.update({"mode": "kb_miss", "answer": None, "needs_llm": True,
                    "reason": result["reason"] + "，建议交给大模型或人工"})
        return out

    # 兜底：置信度不够但槽位足以算出答案时，先给答案并标待复核
    if config.DATASET.startswith("enterprise") and slots:
        if routing.policy().get(cls["intent"], {}).get("action") == "kb":
            from app import enterprise_answer

            res = enterprise_answer.answer(cls["intent"], slots, text)
            if res["ok"]:
                out.update({"mode": "computed", "answer": res["answer"],
                            "computed": res.get("computed"), "source": res["source"],
                            "needs_llm": False, "needs_review": True})
                return out

    out.update({"mode": "llm", "answer": None, "needs_llm": True,
                "reason": "该意图需要生成式回答，交给大模型"})
    return out


@app.post("/intent")
def intent(body: IntentIn) -> dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能是空白字符")
    try:
        result = predictor.predict(text, top_k=body.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    coverage, unseen_run = _detect_coverage(text)
    result["routing"] = routing.decide(result["intent"], result["confidence"],
                                       coverage, unseen_run)
    return result


# ---------------- 可视化面板 ----------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    path = config.PROJ / "reports" / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="先运行 python data/build_dashboard.py")
    return HTMLResponse(path.read_text(encoding="utf-8"))


DOMAIN_HINTS = {"餐馆": ("餐馆", "餐厅", "饭店", "吃饭", "菜"),
                "酒店": ("酒店", "宾馆", "住宿", "房"),
                "景点": ("景点", "公园", "景区", "玩", "门票")}


@app.post("/api/gate")
def api_gate(body: IntentIn) -> dict[str, Any]:
    """面板里的"在线试一条"：门控判定 + 代码作答。"""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能是空白字符")

    # 优先用 4 模型集成（一致 + 置信度阈值）；模型缺失时退回单模型
    try:
        from app.ensemble_gate import gate as ensemble_gate

        decision = ensemble_gate().decide(text)
        route = decision["route"]
        cls = {"confidence": decision["min_p_code"]}
        out: dict[str, Any] = {"route": route, "confidence": decision["min_p_code"],
                               "why": decision["why"], "votes": decision["votes"],
                               "code_answer": None}
    except Exception as exc:
        cls = predictor.predict(text, top_k=1)
        route = "code" if int(cls["intent"]) == 0 else "llm"
        out = {"route": route, "confidence": cls["confidence"],
               "why": f"单模型门控（集成不可用：{exc}）", "code_answer": None}

    if route == "code":
        domain = next((d for d, ws in DOMAIN_HINTS.items() if any(w in text for w in ws)), None)
        if domain is None and len(text) <= 12:
            out.update({"route": "template", "why": "无明确领域，走固定话术",
                        "code_answer": "您好，我可以帮您查餐馆、酒店、景点。"})
            return out
        try:
            res = executor.execute(domain or "餐馆", text, top_k=2)
            out["code_answer"] = res.get("answer") or res.get("reason")
        except Exception as exc:
            out["code_answer"] = f"查库失败：{exc}"
    return out
