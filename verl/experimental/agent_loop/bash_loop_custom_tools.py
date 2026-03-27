# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Bash Loop with Custom Tool Format Support.

Extends FusionAgentLoop by replacing hardcoded <bash>...</bash> extraction
with the ToolParser registry, so each model uses its native tool calling format
(Qwen3 JSON, Qwen3.5 XML, etc.) via apply_chat_template(tools=...).

When no parser is configured or auto-detected, falls back to
extract_bash_command() for backward compatibility.
"""
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.fusion_agent_loop import FusionAgentLoop
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Ordered list of (substring, parser_name) for auto-detection from model path.
# More specific patterns MUST come first: "qwen3.5" before "qwen3", since
# "Qwen/Qwen3.5-4B" contains both "qwen3.5" and "qwen3".
_MODEL_TO_PARSER = [
    ("qwen3.5", "qwen3_coder"),
    ("qwen3-coder", "qwen3_coder"),
    ("qwen3", "hermes"),
    ("qwen2", "hermes"),
    ("gpt-oss", "gpt-oss"),
    # DeepSeek uses hermes-style JSON tool calls inside <tool_call> tags.
    # Its chat template does NOT support `tools=`, so manual injection is needed.
    ("deepseek", "hermes"),
]

# Models whose chat template ignores the `tools` parameter entirely.
# For these, tool definitions must be manually injected into the system prompt.
_MODELS_WITHOUT_TEMPLATE_TOOLS = ["deepseek"]

BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command in the sandbox and return stdout/stderr",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run",
                }
            },
            "required": ["command"],
        },
    },
}


def _infer_tool_parser_name(model_path: str) -> Optional[str]:
    """Infer tool parser name from model path using ordered substring matching.

    Returns the parser name if a known model pattern is found, else None.
    """
    model_lower = model_path.lower()
    for substring, parser_name in _MODEL_TO_PARSER:
        if substring in model_lower:
            return parser_name
    return None


@register("bash_loop_custom_tools")
class BashLoopCustomTools(FusionAgentLoop):
    """
    Stateful agent loop with multi-format tool call support.

    Extends FusionAgentLoop with these changes:
    - Injects tool definitions via apply_chat_template(tools=...)
    - Uses ToolParser.extract_tool_calls() instead of extract_bash_command()
    - Attaches parsed tool_calls to assistant messages for correct template replay
    - Auto-detects the parser from the model path when not explicitly configured
    """

    _use_overlay = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine tool parser: explicit config > auto-detect > None (legacy).
        # Set multi_turn.format to null in your config to trigger auto-detection.
        multi_turn = self.config.actor_rollout_ref.rollout.get("multi_turn", {})
        tool_parser_name = multi_turn.get("format", None) if multi_turn else None
        if tool_parser_name is None:
            model_path = self.config.actor_rollout_ref.model.path
            tool_parser_name = _infer_tool_parser_name(model_path)
            if tool_parser_name:
                logger.info(
                    f"Auto-detected tool parser '{tool_parser_name}' "
                    f"from model path '{model_path}'"
                )

        if tool_parser_name:
            self.tool_parser = ToolParser.get_tool_parser(tool_parser_name, self.tokenizer)
        else:
            self.tool_parser = None

        self.bash_tool_schema = BASH_TOOL_SCHEMA

    def _needs_manual_tool_injection(self) -> bool:
        """Check if the model's chat template ignores the `tools` parameter."""
        model_lower = self.config.actor_rollout_ref.model.path.lower()
        return any(m in model_lower for m in _MODELS_WITHOUT_TEMPLATE_TOOLS)

    def _build_apply_kwargs(self, tool_schemas, **extra):
        """Build kwargs for apply_chat_template, merging tool_schemas safely.

        Avoids TypeError if apply_chat_template_kwargs already contains 'tools'.
        Only passes ``tools`` when it is not None, so legacy/manual-injection
        paths behave identically to the original FusionAgentLoop (no ``tools``
        key at all).
        """
        apply_kwargs = dict(self.apply_chat_template_kwargs)
        if tool_schemas is not None:
            apply_kwargs["tools"] = tool_schemas
        else:
            apply_kwargs.pop("tools", None)
        apply_kwargs.update(extra)
        return apply_kwargs

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one episode of the agent loop with a stateful session."""

        # Parse tools kwargs
        assert "tools_kwargs" in kwargs
        self.tools_kwargs = json.loads(kwargs["tools_kwargs"])
        assert "files_dict" in self.tools_kwargs, f"{self.tools_kwargs=}"
        self.files_to_fetch = self.tools_kwargs.get("files_to_fetch", [])
        startup_commands = self.tools_kwargs.get("startup_commands", [])
        files_dict = self.tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"{files_dict=}"
        files = self.flatten_structure(files_dict)

        # Parse extra files (placed at absolute paths outside working directory)
        extra_files_dict = self.tools_kwargs.get("extra_files_dict", {})
        if extra_files_dict:
            if isinstance(extra_files_dict, dict):
                extra_files = extra_files_dict
            elif isinstance(extra_files_dict, list):
                extra_files = self.flatten_structure(extra_files_dict)
            else:
                extra_files = {}
        else:
            extra_files = {}

        # Create a new session for this episode
        session_id = uuid4().hex
        metrics = {}

        with simple_timer("session_create", metrics):
            try:
                self.current_session_id = await self.session_client.create_session(
                    session_id=session_id,
                    files=files,
                    extra_files=extra_files,
                    startup_commands=startup_commands,
                )
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise

        try:
            messages = list(kwargs["raw_prompt"])
            metrics["tool_calls_count"] = 0
            request_id = uuid4().hex
            num_turns = 0
            multi_turn_cfg = self.config.actor_rollout_ref.rollout.get("multi_turn", {})
            max_num_turns = multi_turn_cfg.get(
                "max_assistant_turns", 5
            ) if multi_turn_cfg else 5
            if max_num_turns is None:
                max_num_turns = 5
            mask = list()

            # --- Modification A/B: Determine tool_schemas for apply_chat_template ---
            if self.tool_parser:
                if self._needs_manual_tool_injection():
                    # Template ignores `tools`; inject definitions into system prompt.
                    # Deep-copy messages to avoid mutating the caller's data.
                    messages = [dict(msg) for msg in messages]
                    tool_description = json.dumps(self.bash_tool_schema, indent=2)
                    injection = (
                        "\n\n# Tools\n\nYou have access to the following tool:\n"
                        + tool_description
                    )
                    injected = False
                    for msg in messages:
                        if msg["role"] == "system":
                            msg["content"] = msg["content"] + injection
                            injected = True
                            break
                    if not injected:
                        messages.insert(
                            0, {"role": "system", "content": injection.lstrip("\n")}
                        )
                    tool_schemas = None
                else:
                    tool_schemas = [self.bash_tool_schema]

                tool_schema_objects = [
                    OpenAIFunctionToolSchema.model_validate(self.bash_tool_schema)
                ]
            else:
                tool_schemas = None
                tool_schema_objects = None

            conversation_messages = [dict(msg) for msg in messages]

            apply_kwargs = self._build_apply_kwargs(
                tool_schemas, add_generation_prompt=True, tokenize=True
            )
            raw_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, **apply_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(raw_prompt_ids)
            curr_input = list(prompt_ids)
            all_output_with_tool = list()
            all_log_probs = list()
            fetched_files = self._EMPTY_FILES

            with simple_timer("generate_sequences_all_turns", metrics):
                while num_turns < max_num_turns:
                    assert len(curr_input) > 0
                    assert isinstance(curr_input, list)
                    assert isinstance(curr_input[-1], int)

                    with simple_timer("generate_sequences", metrics):
                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=curr_input,
                            sampling_params=sampling_params,
                        )
                    assert isinstance(output, TokenOutput), (
                        f"{type(output)=}, {output=}"
                    )
                    assert isinstance(output.token_ids, list)
                    assert isinstance(output.token_ids[0], int)

                    all_output_with_tool += output.token_ids
                    mask += [1] * len(output.token_ids)
                    if output.log_probs is not None and all_log_probs is not None:
                        assert len(output.log_probs) == len(output.token_ids), (
                            f"log_probs length {len(output.log_probs)} != "
                            f"token_ids length {len(output.token_ids)}"
                        )
                        all_log_probs += output.log_probs
                    else:
                        all_log_probs = None

                    decoded_output = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(output.token_ids)
                    )

                    # --- Modification C: Extract bash command via tool parser or legacy ---
                    cmd = None
                    function_calls = []
                    content_text = None
                    if self.tool_parser:
                        try:
                            content_text, function_calls = (
                                await self.tool_parser.extract_tool_calls(
                                    output.token_ids, tool_schema_objects
                                )
                            )
                        except Exception as e:
                            logger.error(
                                f"Tool parser failed, falling back to legacy: {e}"
                            )
                            content_text = decoded_output
                            function_calls = []

                        for fc in function_calls:
                            if fc.name == "bash":
                                try:
                                    args = json.loads(fc.arguments)
                                    cmd = args.get("command")
                                except json.JSONDecodeError:
                                    cmd = fc.arguments
                                break

                        # Fallback: if tool parser found no calls, try legacy
                        # <bash>...</bash> extraction. This handles models that
                        # learned the legacy format from training data.
                        if cmd is None and not function_calls:
                            cmd = await self.loop.run_in_executor(
                                None,
                                lambda: self.extract_bash_command(decoded_output),
                            )
                    else:
                        cmd = await self.loop.run_in_executor(
                            None,
                            lambda: self.extract_bash_command(decoded_output),
                        )

                    # --- Modification D: Attach tool_calls to assistant message ---
                    # When tool_calls are present, use content_text (tool call
                    # tags stripped) so the template doesn't render them twice.
                    # Only attach bash tool_calls (filter out hallucinated names).
                    assistant_msg = {
                        "role": "assistant",
                        "content": content_text if function_calls else decoded_output,
                    }
                    if function_calls:
                        tool_calls_list = []
                        for fc in function_calls:
                            # Keep arguments as the original JSON string.
                            # Templates that need a dict (Qwen3.5) use tojson/items
                            # which handles both; templates that expect a string
                            # (Hermes, DeepSeek) would double-serialize a dict.
                            tool_calls_list.append({
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": fc.arguments,
                                },
                            })
                        assistant_msg["tool_calls"] = tool_calls_list
                    conversation_messages.append(assistant_msg)

                    # Exit if no command
                    if cmd is None:
                        break

                    curr_input += output.token_ids

                    # Check for dangerous commands
                    is_dangerous = any(
                        pattern in cmd for pattern in self._DANGEROUS_PATTERNS
                    )

                    if is_dangerous:
                        first_word = cmd.split()[0] if cmd.strip() else "command"
                        cmd_output = (
                            f"bash: {first_word}: Operation not permitted"
                        )
                        fetched_files = self._EMPTY_FILES
                    elif self.has_junk_artifacts(cmd):
                        cmd_output = "Command contains invalid artifacts"
                        fetched_files = self._EMPTY_FILES
                    else:
                        with simple_timer("tool_calls", metrics):
                            cmd_output, fetched_files = (
                                await self.execute_agent_command(cmd)
                            )
                        metrics["tool_calls_count"] = (
                            metrics.get("tool_calls_count", 0) + 1
                        )

                    tokens_used = len(curr_input)
                    total_budget = self.prompt_length + self.response_length
                    tokens_remaining = max(total_budget - tokens_used, 256)
                    cmd_output = self.truncate_to_token_budget(
                        cmd_output, tokens_remaining
                    )

                    tool_msg = {"role": "tool", "content": cmd_output}

                    # Tokenize tool message via full-conversation delta.
                    # Pass tools= consistently so the delta only captures tool
                    # message tokens, not tool definition changes.
                    conv_before = list(conversation_messages)
                    before_kwargs = self._build_apply_kwargs(
                        tool_schemas,
                        add_generation_prompt=False,
                        tokenize=True,
                    )
                    ids_before = normalize_token_ids(
                        await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                conv_before, **before_kwargs,
                            ),
                        )
                    )
                    conversation_messages.append(tool_msg)
                    conv_after = list(conversation_messages)
                    after_kwargs = self._build_apply_kwargs(
                        tool_schemas,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                    ids_after = normalize_token_ids(
                        await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                conv_after, **after_kwargs,
                            ),
                        )
                    )
                    cmd_message_ids = ids_after[len(ids_before):]
                    curr_input += cmd_message_ids
                    all_output_with_tool += cmd_message_ids
                    if all_log_probs is not None:
                        all_log_probs += [0.0] * len(cmd_message_ids)
                    mask += [0] * len(cmd_message_ids)

                    if len(mask) >= self.response_length:
                        break

                    num_turns += 1

            # Prepare output
            assert len(mask) == len(all_output_with_tool), (
                f"Length mismatch: mask={len(mask)}, "
                f"output={len(all_output_with_tool)}"
            )
            if all_log_probs is not None:
                assert len(all_log_probs) == len(all_output_with_tool), (
                    f"Length mismatch: log_probs={len(all_log_probs)}, "
                    f"output={len(all_output_with_tool)}"
                )
                all_log_probs = all_log_probs[: self.response_length]
            mask = mask[: self.response_length]
            all_output_with_tool = all_output_with_tool[: self.response_length]
            assert len(mask) == len(all_output_with_tool)
            if all_log_probs is not None:
                assert len(all_log_probs) == len(all_output_with_tool)

            # Final file fetch
            if self.files_to_fetch and self.current_session_id:
                with simple_timer("final_file_fetch", metrics):
                    try:
                        final_result = await self.session_client.run_command(
                            session_id=self.current_session_id,
                            command="true",
                            timeout=5,
                            fetch_files=self.files_to_fetch,
                        )
                        final_raw_files = final_result.get("files", {})
                        fetched_files = self.decode_fetched_files(final_raw_files)
                    except Exception as e:
                        logger.warning(f"Final file fetch failed: {e}")

            assert isinstance(fetched_files, np.ndarray)

            return AgentLoopOutput(
                prompt_ids=prompt_ids[: self.prompt_length],
                response_ids=all_output_with_tool[: self.response_length],
                response_mask=mask[: self.response_length],
                response_logprobs=all_log_probs if all_log_probs else None,
                multi_modal_data={},
                num_turns=num_turns,
                metrics=metrics,
                extra_fields=dict(
                    fetched_files=fetched_files,
                    messages=conversation_messages,
                ),
            )

        finally:
            if self.current_session_id:
                with simple_timer("session_destroy", metrics):
                    await self.session_client.destroy_session(
                        self.current_session_id
                    )
                self.current_session_id = None
            if self.session_client:
                with simple_timer("session_close_http", metrics):
                    await self.session_client.close()


@register("bash_loop_custom_tools_overlay")
class BashLoopCustomToolsOverlay(BashLoopCustomTools):
    """
    BashLoopCustomTools variant that uses OverlayFS-isolated sessions.

    Identical behavior but routes session API calls through /overlay-session/
    endpoints for filesystem isolation between sessions.
    """

    _use_overlay = True
