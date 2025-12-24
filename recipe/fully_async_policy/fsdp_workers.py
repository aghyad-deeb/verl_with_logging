# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import os
import asyncio
from dataclasses import dataclass

import torch
import torch.distributed
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from recipe.fully_async_policy.fsdp2_utils import fsdp2_sharded_load_from_cpu, fsdp2_sharded_save_to_cpu
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    collect_lora_params,
    offload_fsdp_model_to_cpu,
    replace_lora_wrapper,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


def get_inference_model(rollout):
    """
    get models according to different types of inference_engine
    Args:
        rollout: rollout object
    Returns:
        model: model object
    """
    inference_engine = rollout.inference_engine
    if hasattr(inference_engine, "llm_engine"):
        inference_model = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    elif hasattr(inference_engine, "worker"):
        inference_model = inference_engine.worker.model_runner.model
    else:
        raise AttributeError(
            f"Unsupported inference_engine type: {type(inference_engine)}. "
            f"Expected LLM (with llm_engine attribute) or WorkerWrapperBase (with worker attribute)."
        )
    return inference_model


class DetachNcclSync(AsyncActorRolloutRefWorker):
    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, peft_config=None, sync_base_if_lora=False):
    # async def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        #assert hasattr(self, "_weights_info") and self._weights_info is not None

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if not peft_config or sync_base_if_lora:
            assert hasattr(self, "_base_weights_info") and self._weights_info is not None
            print(f'[First if in sync_rolloutweights] {sync_base_if_lora=} {peft_config is None=}')
            params, peft_config = self._get_actor_params(get_base_params=True) if self._is_actor else (None, None)
            if self._is_rollout:
                inference_model = get_inference_model(self.rollout)

                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)
            for key, shape, dtype in self._base_weights_info:
                tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                if self._is_actor:
                    assert key in params
                    origin_data = params[key]
                    if hasattr(origin_data, "full_tensor"):
                        origin_data = origin_data.full_tensor()
                    if torch.distributed.get_rank() == 0:
                        tensor.copy_(origin_data)
                from ray.util.collective import collective

                collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
                if self._is_rollout:
                    inference_model.load_weights([(key, tensor)])

            if self._is_actor and self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            get_torch_device().empty_cache()
        else:
            assert hasattr(self, "_lora_weights_info") and self._weights_info is not None
            print(f"\n`sync_rollout_weights`. In else. {self._is_actor=}, {self._is_rollout=} \n")
            per_tensor_param_items, peft_config_from_actor = self._get_actor_params(get_base_params=False) if self._is_actor else (None, None)
            per_tensor_param = dict(per_tensor_param_items) if self._is_actor else None
            print(f"after per_tensor_param. {self._is_actor=}, {self._is_rollout=}")
            # assert isinstance(per_tensor_param, dict)
            collected_per_tensor_param = dict()
            for key, shape, dtype in self._weights_info:
                tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                if self._is_actor:
                    assert key in per_tensor_param, f"{key=}, {list(dict(per_tensor_param).keys())=}"
                    origin_data = per_tensor_param[key]
                    if hasattr(origin_data, "full_tensor"):
                        origin_data = origin_data.full_tensor()
                    if torch.distributed.get_rank() == 0:
                        tensor.copy_(origin_data)
                from ray.util.collective import collective

                collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
                if self._is_rollout:
                    collected_per_tensor_param[key] = tensor
            
            print(f"after per_tensor_param {self._is_actor=}, {self._is_rollout=}")
            if self._is_rollout:
                # print(f"\n\n\n\n\nin self._is_rollout: {collected_per_tensor_param=}\n\n\n\n\n")
                # try:
                #     loop = asyncio.get_event_loop()
                # except RuntimeError:
                #     loop = asyncio.new_event_loop()
                #     asyncio.set_event_loop(loop)
                #! I'm worried about this update
                print(f"just before run_until_complete")
                # @dataclass
                # class DummyConfig:
                #     dummy_val: bool
                # peft_config = DummyConfig(dummy_val=True)
                # loop.run_until_complete(self.rollout.update_weights(collected_per_tensor_param.items(), peft_config=peft_config, base_sync_done=True))
                self.rollout.update_weights_sync(collected_per_tensor_param.items(), peft_config=peft_config, base_sync_done=True)
            
        get_torch_device().empty_cache()
            
            


class DetachActorWorker(DetachNcclSync):
    def _get_actor_params(self, get_base_params=False):
        #! It doesnt seem to handle lora?
        assert self._is_actor
        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        returning_lora = False
        print(f"\n\n{self.config.model.lora_rank=}, {peft_model=}\n\n")
        if hasattr(peft_model, "peft_config"):
            print(f"_get_actor_params. In has attribute if")
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=not get_base_params,
            )
            if get_base_params:
                print(f"\n\n[in get_base_if_lora]\n\n")
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:
                returning_lora = True
                print(f"\n\n[not in get_base_if_lora]\n\n")
        else:
            print(f"_get_actor_params. In has attribute else")
            params = self.actor_module_fsdp.state_dict()
        # params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        # per_tensor_param = params.items() if isinstance(params, dict) else params
        if returning_lora:
            return params, peft_config
        else:
            per_tensor_param = params.items() if isinstance(params, dict) else params 
            return per_tensor_param, peft_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        # if hasattr(self, "_weights_info"):
        #     return self._weights_info
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        base_params, peft_config = self._get_actor_params(get_base_params=False)
        lora_params, peft_config = self._get_actor_params(get_base_params=True)
        ret_base = []
        base_params = dict(base_params)
        for key, tensor in base_params.items():
            ret_base.append((key, tensor.size(), tensor.dtype))
        self._base_weights_info = ret_base

        ret_lora = []
        lora_params = dict(lora_params)
        for key, tensor in lora_params.items():
            ret_lora.append((key, tensor.size(), tensor.dtype))
        self._lora_weights_info = ret_lora
        # ret = []
        # params = dict(params)
        # for key, tensor in params.items():
        #     ret.append((key, tensor.size(), tensor.dtype))
        # self._weights_info = ret
        return ret_base, ret_lora, peft_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}
        self.cpu_saved_models[n] = fsdp2_sharded_save_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        if n in self.cpu_saved_models:
            cpu_sharded_state, global_spec = self.cpu_saved_models[n]
            fsdp2_sharded_load_from_cpu(self.actor_module_fsdp, cpu_sharded_state, global_spec)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        if n in self.cpu_saved_models:
            del self.cpu_saved_models[n]


class DetachAsyncRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        ActorRolloutRefWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, base_weights_info, lora_weights_info):
        assert self._is_rollout
        self._base_weights_info = base_weights_info
        self._lora_weights_info = lora_weights_info
