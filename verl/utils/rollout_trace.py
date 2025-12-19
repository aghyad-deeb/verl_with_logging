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
    """Thread-safe buffer for Inspect AI samples."""
    
    def __init__(self, s3_bucket: str, s3_prefix: str, flush_interval: int = 50):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.flush_interval = flush_interval
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._sample_counter = 0
        self._date = datetime.now().strftime('%Y-%m-%d')
        
    def add_sample(self, messages: List[Dict[str, str]], attributes: Dict[str, Any]):
        """Add a sample to the buffer."""
        with self._lock:
            self._sample_counter += 1
            self._samples.append({
                "id": self._sample_counter,
                "messages": messages,
                "attributes": attributes,
                "timestamp": datetime.now().isoformat(),
            })
            
            if len(self._samples) >= self.flush_interval:
                self._flush_internal()
    
    def _flush_internal(self):
        """Internal flush - must be called with lock held."""
        if not self._samples:
            return
            
        try:
            from inspect_ai.log import EvalLog, EvalSpec, EvalSample, EvalPlan, EvalResults, EvalStats, write_eval_log
            from inspect_ai.log._log import EvalDataset, EvalConfig, EvalScore, EvalMetric
            from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
            from inspect_ai.model._model_output import ModelOutput, ModelUsage, ChatCompletionChoice
            from inspect_ai.model._generate_config import GenerateConfig
            from inspect_ai.scorer._metric import Score
            import shortuuid
            
            eval_id = shortuuid.uuid()
            run_id = shortuuid.uuid()
            config = RolloutTraceConfig.get_instance()
            
            # Convert buffered samples to EvalSamples
            eval_samples = []
            for sample_data in self._samples:
                messages = sample_data["messages"]
                attrs = sample_data["attributes"]
                
                # Convert messages to Inspect format
                inspect_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        inspect_messages.append(ChatMessageSystem(content=content))
                    elif role == "user":
                        inspect_messages.append(ChatMessageUser(content=content))
                    elif role == "assistant":
                        inspect_messages.append(ChatMessageAssistant(content=content))
                    elif role == "tool":
                        inspect_messages.append(ChatMessageTool(content=content, function="bash"))
                
                # Get last assistant message
                last_assistant = ""
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        last_assistant = msg.get("content", "")
                        break
                
                eval_sample = EvalSample(
                    id=sample_data["id"],
                    epoch=1,
                    input=inspect_messages[:2] if len(inspect_messages) >= 2 else inspect_messages,
                    target="rollout",
                    messages=inspect_messages,
                    output=ModelOutput(
                        model=config.project_name or "unknown",
                        choices=[ChatCompletionChoice(
                            message=ChatMessageAssistant(content=last_assistant),
                            stop_reason="stop"
                        )] if last_assistant else [],
                        completion=last_assistant,
                        usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    ),
                    scores={
                        "step": Score(value=attrs.get("step", 0)),
                        "sample_index": Score(value=attrs.get("sample_index", 0)),
                    },
                    metadata={
                        **attrs,
                        "timestamp": sample_data["timestamp"],
                    },
                    store={},
                    events=[],
                    model_usage={},
                )
                eval_samples.append(eval_sample)
            
            # Create EvalSpec
            eval_spec = EvalSpec(
                eval_id=eval_id,
                run_id=run_id,
                created=self._date,
                task=config.experiment_name or "rollout_trace",
                task_id=config.experiment_name or "rollout_trace",
                task_version=1,
                task_attribs={},
                task_args={},
                task_args_passed={},
                dataset=EvalDataset(name="rollout", samples=len(eval_samples), sample_ids=[s.id for s in eval_samples]),
                model=config.project_name or "unknown",
                model_generate_config=GenerateConfig(),
                model_args={},
                config=EvalConfig(),
                packages={},
                metadata={
                    "project_name": config.project_name,
                    "experiment_name": config.experiment_name,
                },
                scorers=[],
            )
            
            now = datetime.now().isoformat()
            eval_log = EvalLog(
                version=2,
                status="success",
                eval=eval_spec,
                plan=EvalPlan(name="rollout_trace", steps=[], config=GenerateConfig()),
                results=EvalResults(total_samples=len(eval_samples), completed_samples=len(eval_samples), scores=[]),
                stats=EvalStats(started_at=self._date, completed_at=now, model_usage={}),
                samples=eval_samples,
            )
            
            # Write to S3
            filename = f"{config.experiment_name}_{self._date}_{eval_id[:8]}.eval"
            s3_path = f"s3://{self.s3_bucket}/logs/{self.s3_prefix}/{self._date}/{filename}"
            write_eval_log(eval_log, s3_path)
            
            self._samples.clear()
            
        except Exception as e:
            print(f"[Inspect Logging] Warning: Failed to flush samples: {e}")
    
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
    tracing backends like Weave, MLflow, and Inspect AI.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', 'inspect', or None).
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
    def reset(cls):
        if cls._instance and cls._instance._inspect_buffer:
            cls._instance._inspect_buffer.finalize()
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
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
    elif backend == "inspect":
        # Store attributes in context variable for Inspect backend
        token = _inspect_attributes.set(attributes)
        try:
            yield
        finally:
            _inspect_attributes.reset(token)
    else:
        yield


def rollout_trace_op(func):
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
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
            
            # Get current attributes from context
            attributes = _inspect_attributes.get().copy()
            
            # Try to extract messages from result or inputs
            messages = []
            
            # Check if result has messages (from fusion_agent_loop)
            if hasattr(result, "extra_fields") and isinstance(result.extra_fields, dict):
                messages = result.extra_fields.get("messages", [])
            
            # If no messages, try to construct from prompt/response text
            if not messages:
                # Try to get raw_prompt from inputs
                raw_prompt = inputs.get("raw_prompt", [])
                if raw_prompt:
                    for msg in raw_prompt:
                        if isinstance(msg, dict):
                            messages.append(msg)
                
                # Add response if we can decode it
                if enable_token2text and hasattr(self, "tokenizer"):
                    if hasattr(result, "response_ids"):
                        loop = asyncio.get_running_loop()
                        response_text = await loop.run_in_executor(None, self.tokenizer.decode, result.response_ids)
                        messages.append({"role": "assistant", "content": response_text})
            
            # Add to buffer if we have messages
            if messages:
                buffer = RolloutTraceConfig.get_inspect_buffer()
                if buffer:
                    buffer.add_sample(messages, attributes)
            
            return result

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
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
            
            # Get current attributes from context
            attributes = _inspect_attributes.get().copy()
            
            # Try to extract messages from result or inputs
            messages = []
            
            # Check if result has messages (from fusion_agent_loop)
            if hasattr(result, "extra_fields") and isinstance(result.extra_fields, dict):
                messages = result.extra_fields.get("messages", [])
            
            # If no messages, try to construct from raw_prompt
            if not messages:
                raw_prompt = inputs.get("raw_prompt", [])
                if raw_prompt:
                    for msg in raw_prompt:
                        if isinstance(msg, dict):
                            messages.append(msg)
            
            # Add to buffer if we have messages
            if messages:
                buffer = RolloutTraceConfig.get_inspect_buffer()
                if buffer:
                    buffer.add_sample(messages, attributes)
            
            return result
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
