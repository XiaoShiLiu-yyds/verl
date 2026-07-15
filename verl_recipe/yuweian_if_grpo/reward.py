#!/usr/bin/env python3
# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""yuweian IF 数据集的 reward 函数（供 verl `reward.custom_reward_function` 加载）。

reward = hard_weight * hard_score + soft_weight * soft_score
  - hard_score  : code_checker（一组 `def check(response_text)`）执行通过率
  - soft_score  : llm_checker（自然语言约束）由 LLM judge 判定通过率；
                  judge 不可用时回退（默认 soft_score = hard_score）。

verl 调用签名：
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs)
  - solution_str : 模型生成的回答（已剥掉 prompt）
  - extra_info   : 预处理时写入的 {id, code_checker[], llm_checker[], num_hard, num_soft}
  - reward_kwargs: 由 `+reward.custom_reward_function.reward_kwargs.X=Y` 注入（值可能为字符串，内部做类型转换）

返回：dict（含 score 与各项诊断，便于 tensorboard 观察奖励信号）。
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger("yuweian_if_grpo.reward")

# 可调参数及其类型（用于把 hydra 注入的字符串值转成正确类型）
_PARAM_TYPES: Dict[str, type] = {
    "hard_weight": float,
    "soft_weight": float,
    "code_checker_timeout": float,
    "judge_api_base": str,
    "judge_api_key": str,
    "judge_model": str,
    "judge_max_tokens": int,
    "judge_temperature": float,
    "judge_timeout": float,
    "judge_fallback": str,
    "print_details": bool,
}

# 默认值（其中 judge_* 优先读环境变量）
def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _defaults() -> Dict[str, Any]:
    return {
        "hard_weight": 0.6,
        "soft_weight": 0.4,
        "code_checker_timeout": 10.0,
        "judge_api_base": _env("IF_JUDGE_API_BASE", "MOCK_API_BASE", "OPENAI_API_BASE"),
        "judge_api_key": _env("IF_JUDGE_API_KEY", "MOCK_API_KEY", "OPENAI_API_KEY", default="dummy_key"),
        "judge_model": _env("IF_JUDGE_MODEL", "MOCK_MODEL_NAME", "OPENAI_MODEL"),
        "judge_max_tokens": 16,
        "judge_temperature": 0.0,
        "judge_timeout": 60.0,
        "judge_fallback": "hard",  # "hard" -> 软分=硬分；"0.5" / "0" -> 常量
        "print_details": False,
    }


def _coerce(cast: type, value: Any) -> Any:
    if value is None:
        return None
    if cast is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
    try:
        if cast is int and isinstance(value, bool):
            return int(value)
        return cast(value)
    except (TypeError, ValueError):
        return None


def _resolve_params(reward_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """合并 默认值 + 注入的 reward_kwargs + 环境变量，并做类型转换。"""
    params = _defaults()
    for name, cast in _PARAM_TYPES.items():
        if name in reward_kwargs:
            coerced = _coerce(cast, reward_kwargs[name])
            if coerced is not None:
                params[name] = coerced
    # judge_* 兜底：注入为空时再回退到环境变量
    if not params["judge_api_base"]:
        params["judge_api_base"] = _defaults()["judge_api_base"]
    if not params["judge_model"]:
        params["judge_model"] = _defaults()["judge_model"]
    return params


# ---------------- 硬约束：exec code_checker ----------------

def _compile_checker(src: str):
    """编译一段 `def check(response_text)` 源码，返回 check 函数；失败返回 None。"""
    # 受信数据（Gemini 生成），但仍在受限命名空间执行 + 全局异常捕获。
    namespace: Dict[str, Any] = {"__name__": "yuweian_code_checker", "__builtins__": __builtins__}
    try:
        exec(compile(src, "<code_checker>", "exec"), namespace)  # noqa: S102
        check = namespace.get("check")
        if not callable(check):
            return None
        return check
    except Exception:  # noqa: BLE001
        return None


def _call_with_timeout(fn: Callable[..., Any], args: tuple, timeout: float) -> Any:
    """单次调用 + 超时；超时抛 FuturesTimeoutError。线程无法强制 kill，但 checker 多为纯 regex，不会真正挂死。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args)
        return fut.result(timeout=timeout)


def _run_code_checkers(code_checkers: List[str], response: str, timeout: float) -> Tuple[int, int, List[str]]:
    """执行所有硬约束。返回 (passed, total, per_checker_pass('1'/'0'/'X'=error))。"""
    total = len(code_checkers)
    if total == 0:
        return 0, 0, []
    marks: List[str] = []
    passed = 0
    for src in code_checkers:
        check = _compile_checker(src)
        if check is None:
            marks.append("X")
            continue
        try:
            result = _call_with_timeout(check, (response,), timeout)
        except FuturesTimeoutError:
            marks.append("X")
            continue
        except Exception:  # noqa: BLE001
            marks.append("X")
            continue
        # 容忍常见真值表达；非真一律判失败
        ok = bool(result) and result is not False and result != 0 and result != "0"
        if ok:
            passed += 1
            marks.append("1")
        else:
            marks.append("0")
    return passed, total, marks


# ---------------- 软约束：LLM judge ----------------

_JUDGE_PROMPT = """You are a strict, deterministic constraint grader.
Decide whether the RESPONSE below satisfies the single CONSTRAINT.
Output EXACTLY one word on the first line — PASS or FAIL — and nothing else.

CONSTRAINT:
{constraint}

RESPONSE:
{response}
"""


def _parse_judge_verdict(text: str) -> bool:
    if not text:
        return False
    first = text.strip().splitlines()[0].strip().upper()
    if first.startswith("PASS"):
        return True
    return False


def _judge_one(constraint: str, response: str, params: Dict[str, Any]) -> Tuple[bool, bool]:
    """判定单条软约束。返回 (passed, had_exception)。had_exception=True 表示调用本身失败（网络/API/解析异常）。"""
    try:
        from openai import OpenAI
    except Exception:  # noqa: BLE001
        return False, True

    try:
        client = OpenAI(
            base_url=params["judge_api_base"],
            api_key=params["judge_api_key"] or "dummy_key",
            timeout=params["judge_timeout"],
        )
        prompt = _JUDGE_PROMPT.format(constraint=constraint, response=response)
        resp = client.chat.completions.create(
            model=params["judge_model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=params["judge_max_tokens"],
            temperature=params["judge_temperature"],
            # 关闭 thinking，避免 Qwen3 思考模型输出冗长内容污染 PASS/FAIL 判定
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = resp.choices[0].message.content or ""
        return _parse_judge_verdict(text), False
    except Exception:  # noqa: BLE001
        return False, True


def _run_llm_checkers(
    llm_checkers: List[str], response: str, params: Dict[str, Any]
) -> Tuple[int, int, int, List[str]]:
    """返回 (passed, total, errors, marks)。errors=发生异常的条数；用于判断 judge 是否整体不可用。"""
    total = len(llm_checkers)
    if total == 0:
        return 0, 0, 0, []
    passed = 0
    errors = 0
    marks: List[str] = []
    for constraint in llm_checkers:
        ok, had_exc = _judge_one(constraint, response, params)
        if had_exc:
            errors += 1
            marks.append("X")
        elif ok:
            passed += 1
            marks.append("1")
        else:
            marks.append("0")
    return passed, total, errors, marks


def _fallback_soft_score(params: Dict[str, Any], hard_score: float) -> float:
    fb = str(params.get("judge_fallback", "hard")).strip().lower()
    if fb in ("hard", "reuse", "same"):
        return hard_score
    try:
        return float(fb)
    except ValueError:
        return hard_score


# ---------------- 主入口 ----------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = "",
    extra_info: Dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> Dict[str, Any]:
    extra_info = extra_info or {}
    code_checkers: List[str] = list(extra_info.get("code_checker") or [])
    llm_checkers: List[str] = list(extra_info.get("llm_checker") or [])
    params = _resolve_params(reward_kwargs)

    response = solution_str or ""

    # --- 硬约束 ---
    hard_passed, hard_total, hard_marks = _run_code_checkers(
        code_checkers, response, params["code_checker_timeout"]
    )
    hard_score = (hard_passed / hard_total) if hard_total > 0 else 1.0

    # --- 软约束 ---
    soft_passed = 0
    soft_total = 0
    soft_errors = 0
    soft_marks: List[str] = []
    judge_used = False
    if soft_total_available := len(llm_checkers):
        soft_passed, soft_total, soft_errors, soft_marks = _run_llm_checkers(llm_checkers, response, params)
        # judge 整体不可用（每条都异常）或未配置端点 -> 回退
        judge_broken = (not params["judge_api_base"]) or (
            soft_total > 0 and soft_errors == soft_total
        )
        if judge_broken:
            soft_score = _fallback_soft_score(params, hard_score)
            judge_used = False
        else:
            soft_score = (soft_passed / soft_total) if soft_total > 0 else 1.0
            judge_used = True
    else:
        # 没有软约束：软分满分，不参与区分（权重仍按 soft_weight 计入，等价于给硬约束加权）
        soft_score = 1.0
        judge_used = False

    hw = float(params["hard_weight"])
    sw = float(params["soft_weight"])
    wsum = hw + sw
    if wsum <= 0:
        hw, sw, wsum = 1.0, 0.0, 1.0
    reward = (hw * hard_score + sw * soft_score) / wsum

    result: Dict[str, Any] = {
        "score": float(reward),
        "hard_score": float(hard_score),
        "soft_score": float(soft_score),
        "hard_passed": hard_passed,
        "hard_total": hard_total,
        "soft_passed": soft_passed,
        "soft_total": soft_total,
        "judge_used": judge_used,
    }

    if params["print_details"]:
        result["hard_marks"] = "".join(hard_marks)
        result["soft_marks"] = "".join(soft_marks)
        logger.info(
            "id=%s reward=%.4f hard=%d/%d soft=%d/%d judge=%s",
            extra_info.get("id"),
            reward,
            hard_passed,
            hard_total,
            soft_passed,
            soft_total,
            judge_used,
        )
    return result


# ---------------- 离线自测 ----------------
def _selftest() -> None:
    """python reward.py 直接运行：用 2 条约束 + 1 个故意失败的回答自测。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    code_checkers = [
        "def check(response_text):\n    return 'SQL' in response_text",  # 希望回答里含 SQL
        "def check(response_text):\n    return response_text.count('```') == 4",
        "def check(response_text):\n    import re\n    return not re.search(r'\\buser\\b', response_text, re.IGNORECASE)",
    ]
    llm_checkers = ["The response must be written in a formal, professional tone."]
    extra = {"id": "selftest", "code_checker": code_checkers, "llm_checker": llm_checkers}

    good = "SQL design. ```json\n{}\n``` ```python\npass\n```"  # 含 SQL + 4 个反引号 + 无 user
    bad = "Hey user, here is something."  # 缺 SQL、反引号数不对、含 user

    for label, resp in [("GOOD", good), ("BAD", bad)]:
        # judge 端点未配置时自动回退到硬分
        out = compute_score("yuweian_if", resp, "", extra)
        print(f"[selftest] {label}: {out}")


if __name__ == "__main__":
    _selftest()
