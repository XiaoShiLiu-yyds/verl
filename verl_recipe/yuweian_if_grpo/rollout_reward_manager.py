#!/usr/bin/env python3
# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""自定义 reward manager：在 verl 内置 NaiveRewardManager 基础上，把每条 rollout 的
input(prompt) / output(response) / reward 打分明细落盘，便于离线分析训练轨迹。

通过 run.sh 里
    reward.reward_manager.source=importlib
    reward.reward_manager.name=RolloutRecordingRewardManager
    reward.reward_manager.module.path=pkg://verl_recipe.yuweian_if_grpo.rollout_reward_manager
加载（verl 用 load_extern_object(module_path, object_name=reward_manager.name) 导入本类，
注意 object_name 取的是顶层 `name` 字段，不是 `module.name`），**无需改动 verl 源码**。打分逻辑与
NaiveRewardManager.run_single 完全一致，只是在拿到分数后多记一条 rollout。

输出（每条 rollout 一个 JSON 文件，无 per-sample 子目录）：
    <ROLLOUT_LOG_DIR>/step_<global_steps>/sample_<pid>_<counter>.json

JSON 内容：
    {
      "step": <int>,
      "data_source": "...",
      "id": <原始样本 id>,
      "input": "<prompt 文本>",
      "output": "<模型生成的 response 文本>",
      "reward": {"score": ..., "hard_score": ..., "soft_score": ..., ...},
      "constraints": {"num_hard": ..., "num_soft": ..., "code_checker": [...], "llm_checker": [...]}
    }

环境变量：
    ROLLOUT_LOG_DIR      根目录；为空则关闭记录
    ROLLOUT_LOG_ENABLED  0/false/no/off 关闭（默认开启）

设计要点：
  * 容错：任何异常都被吞掉，绝不让记录逻辑打断（脆弱的）训练主循环。
  * 非阻塞：文件写丢进 executor 线程池，慢共享盘不会卡住 reward 事件循环
    （避免加剧 judge 排队 / OOM，见 qwen3-32b IF-GRPO 历史 OOM 记录）。
  * 并发安全：reward workers 是各自独立进程；文件名含 pid（跨进程唯一）
    + 单进程内单调计数器，永不冲突。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

logger = logging.getLogger("yuweian_if_grpo.rollout_reward_manager")


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return value


def _json_safe(value: Any) -> Any:
    """可被 JSON 序列化则原样返回，否则退回 str（再不行返回 None）。"""
    try:
        json.dumps(value)
        return value
    except Exception:  # noqa: BLE001
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return None


def _write_rollout_to_disk(file_path: str, record: dict) -> None:
    """原子写一条 rollout 记录；运行在 executor 线程池里（避免卡事件循环）。"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)
    except Exception as e:  # noqa: BLE001
        try:
            logger.warning("[rollout-record] write failed for %s: %s", file_path, e)
        except Exception:  # noqa: BLE001
            pass


class RolloutRecordingRewardManager(NaiveRewardManager):
    """NaiveRewardManager + 离线 rollout 记录。

    仅覆盖 run_single：在原有打分逻辑外，额外解码 prompt 并把
    (input, output, reward 明细, step, 约束) 落盘。打分路径与父类完全一致。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_counter = 0

    def _decode_prompt_and_response(self, data_item, valid_response_ids):
        """解码单条 rollout 的 (prompt, response) 文本。

        在 executor 里跑，避免 tokenization 阻塞事件循环。response 部分沿用原解码；
        prompt 仅用于离线记录，出错则退回空串。
        """
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        prompt_str = ""
        try:
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attn = data_item.batch["attention_mask"]
            valid_prompt_length = int(attn[:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        except Exception:  # noqa: BLE001
            prompt_str = ""
        return prompt_str, response_str

    def _record_rollout(self, data_item, meta_info, prompt_str, response_str, reward_extra_info, score):
        """把一条 rollout（input + output + reward）落盘。best-effort，绝不抛异常。"""
        try:
            log_dir = os.environ.get("ROLLOUT_LOG_DIR", "").strip()
            if not log_dir:
                return
            if os.environ.get("ROLLOUT_LOG_ENABLED", "1").strip().lower() in ("0", "false", "no", "off", ""):
                return

            # global step：v1 在 non_tensor_batch 里带，v0 可能在 meta_info 里。
            step = None
            try:
                gs = data_item.non_tensor_batch.get("global_steps")
                if gs is None:
                    gs = (meta_info or {}).get("global_steps")
                step = int(gs) if gs is not None else None
            except Exception:  # noqa: BLE001
                step = None

            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
            try:
                sample_id = extra_info.get("id")
            except Exception:  # noqa: BLE001
                sample_id = None

            reward_block = {"score": _safe_float(score)}
            for key, value in (reward_extra_info or {}).items():
                reward_block[key] = _json_safe(value)

            record = {
                "step": step,
                "data_source": _json_safe(data_item.non_tensor_batch.get("data_source")),
                "id": _json_safe(sample_id),
                "input": prompt_str,
                "output": response_str,
                "reward": reward_block,
            }
            try:
                code_checkers = extra_info.get("code_checker") or []
                llm_checkers = extra_info.get("llm_checker") or []
                record["constraints"] = {
                    "num_hard": _json_safe(extra_info.get("num_hard", len(code_checkers))),
                    "num_soft": _json_safe(extra_info.get("num_soft", len(llm_checkers))),
                    "code_checker": _json_safe(code_checkers),
                    "llm_checker": _json_safe(llm_checkers),
                }
            except Exception:  # noqa: BLE001
                pass

            # 跨进程(pid) + 单进程内计数器 -> 文件名唯一不冲突
            self._rollout_counter += 1
            step_dir = os.path.join(log_dir, f"step_{step}" if step is not None else "step_unknown")
            file_name = f"sample_{os.getpid()}_{self._rollout_counter:08d}.json"
            file_path = os.path.join(step_dir, file_name)

            # 写盘丢进 executor，慢共享盘不卡事件循环；callable 内部已吞所有异常
            self.loop.run_in_executor(None, _write_rollout_to_disk, file_path, record)
        except Exception as e:  # noqa: BLE001
            try:
                logger.warning("[rollout-record] failed: %s", e)
            except Exception:  # noqa: BLE001
                pass

    async def run_single(self, data: DataProto) -> dict:
        """与父类 NaiveRewardManager.run_single 等价，额外解码 prompt 并落盘记录。"""
        data = data[-1:]  # multi-sequence 时只取最后一条算 reward
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        prompt_str, response_str = await self.loop.run_in_executor(
            None, lambda: self._decode_prompt_and_response(data_item, valid_response_ids)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info: dict = {}
        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        # best-effort 离线 rollout 记录（input + output + reward）
        self._record_rollout(
            data_item=data_item,
            meta_info=data.meta_info,
            prompt_str=prompt_str,
            response_str=response_str,
            reward_extra_info=reward_extra_info,
            score=score,
        )

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
