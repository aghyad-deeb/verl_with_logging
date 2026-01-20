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
<<<<<<< HEAD
import asyncio
from dataclasses import dataclass
=======
import time
>>>>>>> 7522bef0eb5c5761500fa8652e7ed45936f5323d

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

from .checkpoint_engine import CheckpointEngine

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
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_checkpoint_engine(self, rank_offset: int, actor_num: int, rollout_num: int):
        current_rank = torch.distributed.get_rank() + rank_offset
        actor_ranks = list(range(actor_num))
        rollout_ranks = [rank + actor_num for rank in range(rollout_num)]
        assert rank_offset == 0 or rank_offset == actor_num

        self.checkpoint_engine = CheckpointEngine(
            current_rank, actor_ranks, rollout_ranks, self.config.checkpoint_engine.device_buffer_size_M
        )

    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
<<<<<<< HEAD
    def sync_rollout_weights(self, peft_config=None, sync_base_if_lora=False):
    # async def sync_rollout_weights(self):
=======
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
>>>>>>> 7522bef0eb5c5761500fa8652e7ed45936f5323d
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
<<<<<<< HEAD
        if not peft_config or sync_base_if_lora:
            assert hasattr(self, "_base_weights_info") and self._base_weights_info is not None
            print(f'[First if in sync_rolloutweights] {sync_base_if_lora=} {peft_config is None=}')
            params, peft_config = self._get_actor_params(get_base_params=True) if self._is_actor else (None, None)
=======
        params = self._get_actor_params() if self._is_actor else None
        if self._is_rollout:
            inference_model = get_inference_model(self.rollout)

            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)
        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)
            from ray.util.collective import collective

            collective.broadcast(tensor, src_rank=0, group_name=sync_group_name)
>>>>>>> 7522bef0eb5c5761500fa8652e7ed45936f5323d
            if self._is_rollout:
                inference_model = get_inference_model(self.rollout)

                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)
            for key, shape, dtype in self._base_weights_info:
                tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                if self._is_actor:
                    assert key in params, f"{key=}, {list(dict(params).keys())=}, {type(params)}" 
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
            assert hasattr(self, "_lora_weights_info") and self._lora_weights_info is not None
            print(f"\n`sync_rollout_weights`. In else. {self._is_actor=}, {self._is_rollout=} \n")
            per_tensor_param_items, peft_config_from_actor = self._get_actor_params(get_base_params=False) if self._is_actor else (None, None)
            per_tensor_param = dict(per_tensor_param_items) if self._is_actor else None
            print(f"after per_tensor_param. {self._is_actor=}, {self._is_rollout=}")
            # assert isinstance(per_tensor_param, dict)
            collected_per_tensor_param = dict()
            for key, shape, dtype in self._lora_weights_info:
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
            
            

    def cache_actor_weights_to_cpu(self):
        self.cpu_named_params = {}
        if self._is_actor:
            params = self._get_actor_params()
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            for tensor_idx, (key, _, _) in enumerate(self._weights_info):
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()

                if tensor_idx % world_size == local_rank:
                    self.cpu_named_params[key] = origin_data.to("cpu", non_blocking=True)
            get_torch_device().synchronize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_checkpoint(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        # Load model to GPU
        load_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        load_duration = time.time() - load_start_time

        from ray.util.collective import collective

        # Cache actor weights to CPU and measure the time taken
        cache_start_time = time.time()
        self.cache_actor_weights_to_cpu()
        cache_end_time = time.time()
        cache_duration = cache_end_time - cache_start_time

        # Register the cached weights into the checkpoint engine
        self.checkpoint_engine.register_checkpoint(self._weights_info, self.cpu_named_params)
        register_end_time = time.time()
        register_duration = register_end_time - cache_end_time
        self.cpu_named_params = {}

        collective.barrier(group_name=sync_group_name)
        update_start_time = time.time()

        inference_model = None
        if self._is_rollout:
            inference_model = get_inference_model(self.rollout)
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)

        # Update the checkpoint with the inference model and broadcast weights
        self.checkpoint_engine.update_checkpoint(
            inference_model=inference_model,
            group_name=sync_group_name,
            overlap_broadcast_and_consume=self.config.checkpoint_engine.overlap_broadcast_and_consume,
        )

        update_end_time = time.time()
        update_duration = update_end_time - update_start_time

        offload_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        offload_duration = time.time() - offload_start_time

        print(
            f"sync_rollout_weights_by_checkpoint finish!, rank:{torch.distributed.get_rank()},"
            f" is_actor:{self._is_actor}, is_rollout:{self._is_rollout},"
            f" total cost:{update_end_time - cache_start_time} seconds, while cache cost {cache_duration} seconds, "
            f" register cost {register_duration} seconds, update cost {update_duration} seconds"
        )

        if self._is_actor and self._is_offload_param:
            print(
                f"sync_rollout_weights_by_checkpoint load model to gpu cost {load_duration} seconds,"
                f" offload model to cpu cost {offload_duration} seconds"
            )


class DetachActorWorker(DetachNcclSync):
    def _get_actor_params(self, get_base_params=True):
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

        # print(f"\n[In get actor params] {list(dict(params).keys())=}")
        # per_tensor_param = params.items() if isinstance(params, dict) else params
        return params, peft_config
        # if returning_lora:
        #     return params, peft_config
        # else:
        #     per_tensor_param = params.items() if isinstance(params, dict) else params 
        #     return per_tensor_param, peft_config

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
        base_params, peft_config = self._get_actor_params(get_base_params=True)
        lora_params, peft_config = self._get_actor_params(get_base_params=False)
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
    def init_model(self):
        """Override to skip FSDP model loading - rollout only needs vLLM.
        
        The parent class loads both FSDP model AND vLLM on rollout workers,
        but for async training the FSDP model is not needed on rollout nodes
        since weight sync goes directly to vLLM via broadcast.
        This saves ~20GB/GPU for the 80B model.
        """
        from torch.distributed.device_mesh import init_device_mesh
        from verl.workers.fsdp_workers import import_external_libs
        from verl.workers.rollout import get_rollout_class
        from verl.workers.config import HFModelConfig, RolloutConfig
        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.device import get_torch_device
        
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        
        print("[DetachAsyncRolloutWorker] Skipping FSDP model, only building vLLM rollout")
        
        # === Build rollout without FSDP model ===
        # (Copied from _build_rollout, but without the FSDP state_dict setup)
        
        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )

        self.rollout_device_mesh = rollout_device_mesh

        # Register dispatch info
        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model (vLLM)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )

        # Skip FSDP state_dict setup - not needed for async rollout
        # Weight sync goes directly to vLLM via broadcast
        
        # Set base_sync_done for weight loading
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, base_weights_info, lora_weights_info):
        assert self._is_rollout
        #print(f"\n\n{base_weights_info=}\n\n{lora_weights_info=}\n\n")
        self._base_weights_info = base_weights_info
        self._lora_weights_info = lora_weights_info
