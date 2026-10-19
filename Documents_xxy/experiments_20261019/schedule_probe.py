"""Opt-in scheduled-prefill tracing; diagnostic timings are not performance data."""

import json
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class PrefillProbeScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        directory = Path(os.environ["KVEXP_STATS_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self._probe_log = (directory / f"prefill-steps-{os.getpid()}.jsonl").open(
            "a", buffering=1
        )
        self._probe_step = 0

    def _update_after_schedule(self, scheduler_output):
        self._probe_step += 1
        rows = []
        for rid, count in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[rid]
            start = request.num_computed_tokens
            prefill = min(count, max(request.num_prompt_tokens - start, 0))
            if prefill:
                rows.append(
                    {
                        "request_id": rid,
                        "start": start,
                        "prompt_tokens": request.num_prompt_tokens,
                        "scheduled_prefill_tokens": prefill,
                        "remaining_prompt_tokens": max(
                            request.num_prompt_tokens - start - prefill, 0
                        ),
                    }
                )
        super()._update_after_schedule(scheduler_output)
        if rows:
            self._probe_log.write(
                json.dumps(
                    {
                        "step": self._probe_step,
                        "total_scheduled_tokens": (
                            scheduler_output.total_num_scheduled_tokens
                        ),
                        "requests": rows,
                    }
                )
                + "\n"
            )
