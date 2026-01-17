# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import atexit
import contextlib
import functools
import inspect
import os
import threading
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Dict, Any, List

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)
# Context variable to store current trace attributes for Inspect backend
_inspect_attributes: ContextVar[Dict[str, Any]] = ContextVar("_inspect_attributes", default={})


class InspectLogBuffer:
    """Thread-safe buffer for Inspect AI samples. Flushes per training step."""

    def __init__(self, s3_bucket: str, s3_prefix: str, flush_interval: int = 50):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.flush_interval = flush_interval  # kept for fallback, but step-based flush is primary
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._sample_counter = 0
        self._date = datetime.now().strftime('%Y-%m-%d')
        self._current_step: Optional[int] = None

    def add_sample(self, messages: List[Dict[str, str]], attributes: Dict[str, Any]):
        """Add a sample to the buffer. Flushes when step changes."""
        with self._lock:
            sample_step = attributes.get("step")

            # If step changed, flush previous step's samples first
            if self._current_step is not None and sample_step != self._current_step and self._samples:
                self._flush_internal()

            self._current_step = sample_step
            self._sample_counter += 1
            self._samples.append({
                "id": self._sample_counter,
                "messages": messages,
                "attributes": attributes,
                "timestamp": datetime.now().isoformat(),
            })

    def _flush_internal(self):
        """Internal flush - must be called with lock held."""
        if not self._samples:
            return

        try:
            import json
            import io
            import zipfile
            import boto3
            import shortuuid
            import os

            aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
            if aws_key:
                print(f"[Inspect Logging] AWS credentials loaded (key starts with: {aws_key[:5]}...)")
            else:
                print("[Inspect Logging] WARNING: AWS_ACCESS_KEY_ID not found in environment!")
            config = RolloutTraceConfig.get_instance()
            eval_id = shortuuid.uuid()
            run_id = shortuuid.uuid()
            now = datetime.now().isoformat()

            # Build EvalSpec structure
            eval_spec = {
                "eval_id": eval_id,
                "run_id": run_id,
                "created": now,
                "task": config.experiment_name or "rollout_trace",
                "task_id": config.experiment_name or "rollout_trace",
                "task_version": 1,
                "model": config.project_name or "unknown",
                "dataset": {
                    "name": "rollout",
                    "samples": len(self._samples),
                    "sample_ids": [s["id"] for s in self._samples],
                },
                "config": {"epochs": 1},
                "packages": {},
                "metadata": {"project_name": config.project_name, "experiment_name": config.experiment_name},
            }

            # Build EvalPlan structure
            eval_plan = {
                "name": "rollout_trace",
                "steps": [],
                "config": {},
            }

            # _journal/start.json - Required by Inspect AI
            start_json = {
                "version": 2,
                "eval": eval_spec,
                "plan": eval_plan,
            }

            # Build sample summaries
            summaries = []
            for s in self._samples:
                scores = {
                    "step": {"value": s["attributes"].get("step", 0)},
                    "data_source": {"value": s["attributes"].get("data_source", "unknown")},
                }
                if "reward" in s["attributes"]:
                    scores["reward"] = {"value": s["attributes"]["reward"]}
                summaries.append({
                    "id": s["id"],
                    "epoch": 1,
                    "uuid": shortuuid.uuid(),
                    "input": s["messages"][0]["content"][:200] if s["messages"] else "",
                    "target": s["attributes"].get("data_source", "unknown"),
                    "scores": scores,
                })

            # Build header.json (full EvalLog)
            header_json = {
                "version": 2,
                "status": "success",
                "eval": eval_spec,
                "plan": eval_plan,
                "results": {
                    "total_samples": len(self._samples),
                    "completed_samples": len(self._samples),
                },
                "stats": {
                    "started_at": now,
                    "completed_at": now,
                },
            }

            # Include step number and worker PID in filename for easy identification
            # Multiple Ray workers have separate buffers, so we include PID to distinguish them
            import os
            step = self._current_step if self._current_step is not None else "unknown"
            worker_id = os.getpid()
            filename = f"{config.experiment_name}_step{step}_w{worker_id}_{eval_id[:8]}.eval"
            s3_key = f"logs/{self.s3_prefix}/{self._date}/{filename}"

            # Create zip file in memory with proper Inspect AI structure
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
                # 1. _journal/start.json (Required!)
                zf.writestr("_journal/start.json", json.dumps(start_json, default=str))

                # 2. Individual sample files in samples/ directory
                for s in self._samples:
                    # Add id to each message (required by Inspect AI)
                    messages_with_ids = []
                    for idx, msg in enumerate(s["messages"]):
                        msg_with_id = {
                            "id": shortuuid.uuid(),
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                        messages_with_ids.append(msg_with_id)

                    sample_data = {
                        "id": s["id"],
                        "epoch": 1,
                        "uuid": shortuuid.uuid(),
                        "input": s["messages"][0]["content"] if s["messages"] else "",
                        "target": s["attributes"].get("data_source", "unknown"),
                        "messages": messages_with_ids,
                        "output": {
                            "model": config.project_name or "unknown",
                            "choices": [{
                                "message": {
                                    "id": shortuuid.uuid(),
                                    "role": "assistant",
                                    "content": s["messages"][-1]["content"] if s["messages"] else ""
                                },
                                "stop_reason": "stop"
                            }],
                        },
                        "scores": {
                            "step": {"value": s["attributes"].get("step", 0)},
                            "sample_index": {"value": s["attributes"].get("sample_index", 0)},
                            "data_source": {"value": s["attributes"].get("data_source", "unknown")},
                            **( {"reward": {"value": s["attributes"]["reward"]}} if "reward" in s["attributes"] else {}),
                        },
                        "metadata": {**s["attributes"], "timestamp": s["timestamp"]},
                    }
                    zf.writestr(f"samples/{s['id']}_epoch_1.json", json.dumps(sample_data, default=str))

                # 3. summaries.json
                zf.writestr("summaries.json", json.dumps(summaries, default=str))

                # 4. header.json
                zf.writestr("header.json", json.dumps(header_json, default=str))

            # Upload to S3
            s3_client = boto3.client('s3')
            zip_buffer.seek(0)
            s3_client.upload_fileobj(zip_buffer, self.s3_bucket, s3_key)

            print(f"[Inspect Logging] Flushed {len(self._samples)} samples to s3://{self.s3_bucket}/{s3_key}")
            self._samples.clear()

        except Exception as e:
            import traceback
            print(f"[Inspect Logging] Warning: Failed to flush samples: {e}")
            traceback.print_exc()

    def flush(self):
        """Flush buffered samples to S3."""
        with self._lock:
            self._flush_internal()

    def finalize(self):
        """Finalize and flush any remaining samples."""
        self.flush()


class JSONLLogBuffer:
    """Minimal buffer for JSONL logging. Messages stored unaltered."""

    def __init__(self, s3_bucket: str, project_name: str, experiment_name: str):
        self.s3_bucket = s3_bucket
        self.project_name = project_name
        self.experiment_name = experiment_name
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._current_step: Optional[int] = None
        self._date = datetime.now().strftime('%Y-%m-%d')

    def add_sample(self, messages: List[Dict[str, Any]], attributes: Dict[str, Any]):
        """Add sample with messages stored exactly as provided - NO ALTERATION."""
        import uuid
        with self._lock:
            step = attributes.get("step")
            # If step changed, flush previous step's samples first
            if self._current_step is not None and step != self._current_step and self._samples:
                self._flush_internal()
            self._current_step = step
            # Generate unique ID combining step and UUID for easy identification
            sample_id = f"step{step}_{uuid.uuid4().hex[:8]}"
            self._samples.append({
                "id": sample_id,
                "messages": messages,  # Store exactly as received
                "attributes": attributes,
                "timestamp": datetime.now().isoformat(),
            })

    def _flush_internal(self):
        """Write JSONL to S3. Must be called with lock held."""
        if not self._samples:
            return

        try:
            import json
            import boto3

            # Build JSONL content - one JSON object per line
            lines = [json.dumps(s, default=str) for s in self._samples]
            content = "\n".join(lines)

            # Build S3 key path
            step = self._current_step if self._current_step is not None else "unknown"
            pid = os.getpid()
            s3_key = f"logs_jsonl/rollout_traces/{self.project_name}/{self.experiment_name}/{self._date}/step_{step}_w{pid}.jsonl"

            # Upload to S3
            boto3.client('s3').put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=content.encode('utf-8')
            )

            print(f"[JSONL Logging] Flushed {len(self._samples)} samples to s3://{self.s3_bucket}/{s3_key}")
            self._samples.clear()

        except Exception as e:
            import traceback
            print(f"[JSONL Logging] Warning: Failed to flush samples: {e}")
            traceback.print_exc()

    def flush(self):
        """Flush buffered samples to S3."""
        with self._lock:
            self._flush_internal()

    def finalize(self):
        """Finalize and flush any remaining samples."""
        self.flush()


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
    tracing backends like Weave, MLflow, Inspect AI, and JSONL.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', 'inspect', 'jsonl', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: Optional[str] = None
    client: Optional[object] = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: Optional[int] = None
    _inspect_buffer: Optional[InspectLogBuffer] = None
    _jsonl_buffer: Optional[JSONLLogBuffer] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: Optional[int] = None,
        inspect_s3_bucket: str = "rewardseeker",
        inspect_s3_prefix: str = "rollout_traces",
        inspect_flush_interval: int = 50,
    ):
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        elif backend == "inspect":
            config._inspect_buffer = InspectLogBuffer(
                s3_bucket=inspect_s3_bucket,
                s3_prefix=f"{inspect_s3_prefix}/{project_name}/{experiment_name}",
                flush_interval=inspect_flush_interval,
            )
            config.client = config._inspect_buffer
            # Register atexit handler to flush on exit
            atexit.register(config._inspect_buffer.finalize)
        elif backend == "jsonl":
            config._jsonl_buffer = JSONLLogBuffer(
                s3_bucket=inspect_s3_bucket,
                project_name=project_name,
                experiment_name=experiment_name,
            )
            config.client = config._jsonl_buffer
            # Register atexit handler to flush on exit
            atexit.register(config._jsonl_buffer.finalize)
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> Optional[str]:
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> Optional[object]:
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> Optional[bool]:
        return cls.get_instance().token2text

    @classmethod
    def get_inspect_buffer(cls) -> Optional[InspectLogBuffer]:
        return cls.get_instance()._inspect_buffer

    @classmethod
    def get_jsonl_buffer(cls) -> Optional[JSONLLogBuffer]:
        return cls.get_instance()._jsonl_buffer

    @classmethod
    def reset(cls):
        if cls._instance and cls._instance._inspect_buffer:
            cls._instance._inspect_buffer.finalize()
        if cls._instance and cls._instance._jsonl_buffer:
            cls._instance._jsonl_buffer.finalize()
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True,
    data_source=None,
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
        data_source: Data source identifier for the sample.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}
    if backend:
        if sample_index is not None:
            attributes["sample_index"] = sample_index
        if step is not None:
            attributes["step"] = step
        if rollout_n is not None:
            attributes["rollout_n"] = rollout_n
        if data_source is not None:
            attributes["data_source"] = data_source
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    if backend == "weave":
        import weave

        with weave.attributes(attributes):
            yield
    elif backend == "mlflow":
        import mlflow

        with mlflow.start_span(name=name) as span:
            trace_id = span.trace_id
            for key, value in attributes.items():
                mlflow.set_trace_tag(trace_id, str(key), str(value))
            yield
    elif backend == "inspect" or backend == "jsonl":
        # Store attributes in context variable for Inspect/JSONL backend
        token = _inspect_attributes.set(attributes)
        try:
            yield
        finally:
            _inspect_attributes.reset(token)
    else:
        yield


def _get_trace_enabled():
    """Access _trace_enabled via function to avoid closure serialization issues with Ray."""
    return _trace_enabled.get()


def _get_inspect_attributes():
    """Access _inspect_attributes via function to avoid closure serialization issues with Ray."""
    return _inspect_attributes.get()


def rollout_trace_op(func):
    # NOTE: This decorator must NOT capture ContextVar objects directly in closures.
    # Ray cannot serialize ContextVar objects, so we access them through helper functions
    # that are looked up at runtime via the module's global namespace.

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        # Access ContextVar through helper function to avoid serialization issues
        if not _get_trace_enabled():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        async def add_token2text(self, result):
            if hasattr(result, "prompt_ids") and hasattr(self, "tokenizer") and hasattr(self.tokenizer, "decode"):
                _result = vars(result)
                loop = asyncio.get_running_loop()
                if hasattr(result, "prompt_ids"):
                    prompt_text = await loop.run_in_executor(None, self.tokenizer.decode, result.prompt_ids)
                    _result["prompt_text"] = prompt_text

                if hasattr(result, "response_ids"):
                    response_text = await loop.run_in_executor(None, self.tokenizer.decode, result.response_ids)
                    _result["response_text"] = response_text
                return _result
            return result

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await add_token2text(self, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=result)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result

        elif backend == "inspect":
            # Execute the function
            result = await func(self, *args, **kwargs)

            # Only log at _run_agent_loop_inner level (has raw_prompt), not at generate level
            # This avoids double-logging
            raw_prompt = inputs.get("raw_prompt")
            if not raw_prompt:
                return result

            # Get current attributes from context (via helper function)
            attributes = _get_inspect_attributes().copy()

            # Get reward from result if available (from _InternalAgentLoopOutput)
            reward_score = getattr(result, "reward_score", None)
            if reward_score is not None:
                attributes["reward"] = reward_score

            # Check if result has full conversation messages (from agent loops like FusionAgentLoop)
            # This includes all tool/command calls and their outputs
            extra_fields = getattr(result, "extra_fields", {})
            conversation_messages = extra_fields.get("messages") if isinstance(extra_fields, dict) else None

            if conversation_messages and isinstance(conversation_messages, list) and len(conversation_messages) > 0:
                # Use the full conversation which includes tool calls
                messages = []
                for msg in conversation_messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        # Content might be a list (for multimodal), convert to string
                        if isinstance(content, list):
                            content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)
                        messages.append({"role": msg.get("role", "user"), "content": str(content)})
            else:
                # Fallback: reconstruct from raw_prompt + decoded response
                # raw_prompt is a list of message dicts like [{"role": "user", "content": "..."}]
                messages = []
                for msg in raw_prompt:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        # Content might be a list (for multimodal), convert to string
                        if isinstance(content, list):
                            content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)
                        messages.append({"role": msg.get("role", "user"), "content": str(content)})

                # Get tokenizer from self
                tokenizer = getattr(self, "tokenizer", None)

                if enable_token2text and tokenizer:
                    loop = asyncio.get_running_loop()

                    # Get response from response_ids (tensor from _InternalAgentLoopOutput)
                    response_ids = getattr(result, "response_ids", None)
                    if response_ids is not None:
                        # Handle tensor
                        if hasattr(response_ids, "tolist"):
                            ids_list = response_ids[0].tolist()
                        else:
                            ids_list = response_ids
                        response_text = await loop.run_in_executor(
                            None,
                            lambda: tokenizer.decode(ids_list, skip_special_tokens=True)
                        )
                        messages.append({"role": "assistant", "content": str(response_text)})

            # Add to buffer if we have messages
            if messages:
                buffer = RolloutTraceConfig.get_inspect_buffer()
                if buffer:
                    buffer.add_sample(messages, attributes)

            return result

        elif backend == "jsonl":
            # Execute the function
            result = await func(self, *args, **kwargs)

            # Only log at _run_agent_loop_inner level (has raw_prompt), not at generate level
            raw_prompt = inputs.get("raw_prompt")
            if not raw_prompt:
                return result

            # Get current attributes from context (via helper function)
            attributes = _get_inspect_attributes().copy()

            # Get reward from result if available
            reward_score = getattr(result, "reward_score", None)
            if reward_score is not None:
                attributes["reward"] = reward_score

            # Get messages - use conversation_messages if available, else raw_prompt + decoded response
            # CRITICAL: Messages are stored EXACTLY as received - NO ALTERATION
            extra_fields = getattr(result, "extra_fields", {})
            messages = extra_fields.get("messages") if isinstance(extra_fields, dict) else None

            if not messages:
                # Start with raw_prompt (stored as-is)
                messages = list(raw_prompt) if raw_prompt else []

                # Decode response_ids if available to get the full conversation
                tokenizer = getattr(self, "tokenizer", None)
                if tokenizer:
                    response_ids = getattr(result, "response_ids", None)
                    if response_ids is not None:
                        loop = asyncio.get_running_loop()
                        # Handle tensor
                        if hasattr(response_ids, "tolist"):
                            ids_list = response_ids[0].tolist()
                        else:
                            ids_list = response_ids
                        response_text = await loop.run_in_executor(
                            None,
                            lambda: tokenizer.decode(ids_list, skip_special_tokens=True)
                        )
                        messages.append({"role": "assistant", "content": response_text})

            # Add to buffer if we have messages
            if messages:
                buffer = RolloutTraceConfig.get_jsonl_buffer()
                if buffer:
                    buffer.add_sample(messages, attributes)

            return result

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Access ContextVar through helper function to avoid serialization issues
        if not _get_trace_enabled():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        elif backend == "inspect":
            # Execute the function
            result = func(self, *args, **kwargs)

            # Get current attributes from context (via helper function)
            attributes = _get_inspect_attributes().copy()

            # Get reward from result if available
            reward_score = getattr(result, "reward_score", None)
            if reward_score is not None:
                attributes["reward"] = reward_score

            # Check if result has full conversation messages (from agent loops like FusionAgentLoop)
            # This includes all tool/command calls and their outputs
            extra_fields = getattr(result, "extra_fields", {})
            conversation_messages = extra_fields.get("messages") if isinstance(extra_fields, dict) else None

            messages = []
            if conversation_messages and isinstance(conversation_messages, list) and len(conversation_messages) > 0:
                # Use the full conversation which includes tool calls
                for msg in conversation_messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        # Content might be a list (for multimodal), convert to string
                        if isinstance(content, list):
                            content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)
                        messages.append({"role": msg.get("role", "user"), "content": str(content)})
            else:
                # Fallback: try to construct from raw_prompt
                raw_prompt = inputs.get("raw_prompt", [])
                if raw_prompt:
                    for msg in raw_prompt:
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            # Content might be a list (for multimodal), convert to string
                            if isinstance(content, list):
                                content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)
                            messages.append({"role": msg.get("role", "user"), "content": str(content)})

            # Add to buffer if we have messages
            if messages:
                buffer = RolloutTraceConfig.get_inspect_buffer()
                if buffer:
                    buffer.add_sample(messages, attributes)

            return result

        elif backend == "jsonl":
            # Execute the function
            result = func(self, *args, **kwargs)

            # Get current attributes from context (via helper function)
            attributes = _get_inspect_attributes().copy()

            # Get reward from result if available
            reward_score = getattr(result, "reward_score", None)
            if reward_score is not None:
                attributes["reward"] = reward_score

            # Get messages - use conversation_messages if available, else raw_prompt + decoded response
            # CRITICAL: Messages are stored EXACTLY as received - NO ALTERATION
            extra_fields = getattr(result, "extra_fields", {})
            messages = extra_fields.get("messages") if isinstance(extra_fields, dict) else None

            if not messages:
                raw_prompt = inputs.get("raw_prompt", [])
                # Start with raw_prompt (stored as-is)
                messages = list(raw_prompt) if raw_prompt else []

                # Decode response_ids if available to get the full conversation
                tokenizer = getattr(self, "tokenizer", None)
                if tokenizer:
                    response_ids = getattr(result, "response_ids", None)
                    if response_ids is not None:
                        # Handle tensor
                        if hasattr(response_ids, "tolist"):
                            ids_list = response_ids[0].tolist()
                        else:
                            ids_list = response_ids
                        response_text = tokenizer.decode(ids_list, skip_special_tokens=True)
                        messages.append({"role": "assistant", "content": response_text})

            # Add to buffer if we have messages
            if messages:
                buffer = RolloutTraceConfig.get_jsonl_buffer()
                if buffer:
                    buffer.add_sample(messages, attributes)

            return result

        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

