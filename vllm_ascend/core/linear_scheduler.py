##
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/v1/core/sched/scheduler.py
#

from __future__ import annotations

from dataclasses import dataclass, fields
import time
from typing import Type, Union
from collections import deque

from vllm.config import SchedulerConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID


logger = init_logger(__name__)


@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    scheduler_cls: Union[str, Type[object]] = (
        "vllm_ascend.core.linear_scheduler.LinearScheduler")

    @classmethod
    def initialize_from_config(cls, vllm_config: VllmConfig):
        vllm_scheduler_config = vllm_config.scheduler_config
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config) if field.init
        }
        if vllm_scheduler_config.async_scheduling:
            scheduler_config["scheduler_cls"] = (
                "vllm_ascend.core.linear_scheduler.AsyncLinearScheduler")
        else:
            scheduler_config["scheduler_cls"] = (
                "vllm_ascend.core.linear_scheduler.LinearScheduler")
        scheduler_config[
            "max_model_len"] = vllm_config.model_config.max_model_len
        scheduler_config[
            "is_encoder_decoder"] = vllm_config.model_config.is_encoder_decoder
        return cls(**scheduler_config)


class LinearScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_kv_consumer = self.vllm_config.kv_transfer_config and \
                              self.vllm_config.kv_transfer_config.is_kv_consumer
        self.request_first_token: dict[int, dict[str, int]] = {} # [client_index, [request_id, token_id]]

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            if self.is_kv_consumer and request.kv_transfer_params is not None:
                new_token_id = request.kv_transfer_params.get("new_token_id", 0)
                request.prompt_token_ids.append(new_token_id)
                request._all_token_ids.append(new_token_id)
                request.num_prompt_tokens = len(request.prompt_token_ids)
                if request.client_index not in self.request_first_token.keys():
                    self.request_first_token[request.client_index] = {}
                self.request_first_token[request.client_index][request.request_id] = new_token_id
            self.waiting.add_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]
        if self.is_kv_consumer:
            if request.client_index in self.request_first_token.keys() and request.request_id in self.request_first_token[request.client_index].keys():
                self.request_first_token[request.client_index].pop(request.request_id)
        
    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Linear model is not support prefix caching now
            num_computed_tokens = request.num_tokens - 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True
    
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        engine_core_outputs = super().update_from_output(scheduler_output, model_runner_output)
        if self.is_kv_consumer:
            for client_index, client_engine_core_outputs in engine_core_outputs.items():
                if client_index in self.request_first_token.keys():
                    for engine_core_output in client_engine_core_outputs.outputs:
                        if engine_core_output.request_id in self.request_first_token[client_index].keys():
                            engine_core_output.new_token_ids.insert(0, self.request_first_token[client_index][engine_core_output.request_id])
                            self.request_first_token[client_index].pop(engine_core_output.request_id)
        return engine_core_outputs
    

class AsyncLinearScheduler(AsyncScheduler, LinearScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        engine_core_outputs = AsyncScheduler.update_from_output(self, scheduler_output, model_runner_output)
        if self.is_kv_consumer:
            for client_index, client_engine_core_outputs in engine_core_outputs.items():
                for engine_core_output in client_engine_core_outputs.outputs:
                    if client_index in self.request_first_token.keys():
                        if engine_core_output.request_id in self.request_first_token[client_index].keys():
                            engine_core_output.new_token_ids.insert(0, self.request_first_token[client_index][engine_core_output.request_id])
                            self.request_first_token[client_index].pop(engine_core_output.request_id)
        return engine_core_outputs
