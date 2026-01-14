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
import asyncio
import logging
import os
import base64
import requests
import numpy as np
from typing import Any, Optional
from uuid import uuid4

from recipe.fully_async_policy.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop import AgentLoopBase
from verl.experimental.agent_loop.agent_loop import register
from verl.workers.rollout.replica import TokenOutput
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def check_server_running():
    try:
        response = requests.get('http://localhost:60808/health', timeout=2)
        return True
    except:
        try:
            response = requests.post('http://localhost:60808/health', json={
                'code': 'echo "test"',
                'language': 'bash',
                'files': {}
            }, timeout=2)
            return response.status_code == 200
        except Exception as e:
            raise RuntimeError(f"Sandbox server is not running on 'http://localhost:60808/health'. Start it with: docker run -it -p 60808:8080 volcengine/sandbox-fusion:server-20250609") from e

@register("partial_fusion_agent_loop")
class PartialFusionAgentLoop(AgentLoopBase):
    url = 'http://localhost:60808/run_code'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        check_server_running()

    def flatten_structure(self, fs_list, prefix=""):
        files = {}
        for item in fs_list:
            path = f"{prefix}/{item['name']}" if prefix else item['name']
            if item['type'] == 'file':
                files[path] = base64.b64encode(item['content'].encode()).decode()
            else:
                files.update(self.flatten_structure(item['content'], path))
        return files

    def extract_bash_command(self, text, prefix="<bash>", suffix="</bash>"):
        assert isinstance(text, str), f"text must be a string, got {type(text)}"
        assert isinstance(prefix, str), f"prefix must be a string, got {type(prefix)}"
        assert isinstance(suffix, str), f"suffix must be a string, got {type(suffix)}"
        assert len(prefix) > 0, "prefix cannot be empty"
        assert len(suffix) > 0, "suffix cannot be empty"
        eot = "</think>"
        if eot in text:
            text = text.split(eot)[-1]
        # if eot not in s:
        #     return None
        if prefix not in text:
            return None

        after_prefix = text.split(prefix)[-1]
        i = -1
        while suffix not in after_prefix:
            i -= 1
            if len(text.split(prefix)) < abs(i):
                break
            after_prefix = text.split(prefix)[i]

        if suffix not in after_prefix:
            return None

        ret = after_prefix.split(suffix)[0]
        if ret.startswith("\n"):
            ret = ret[1:]
        return ret

    def is_valid_bash_syntax(self, code):
        """Check if command has valid bash syntax using bash -n in sandbox."""
        # Write command to a temp file and validate it (avoids escaping issues)
        script_b64 = base64.b64encode(code.encode()).decode()
        try:
            response = requests.post(self.url, json={
                'code': 'bash -n __validate_cmd.sh',
                'language': 'bash',
                'run_timeout': 1,
                'files': {'__validate_cmd.sh': script_b64},
            }, timeout=5)
            result = response.json()
            # Check both top-level status and run_result return_code
            if result.get("status") != "Success":
                return False
            run_result = result.get("run_result", {})
            return run_result.get("return_code") == 0
        except:
            return False

    def has_junk_artifacts(self, code):
        """Check for markdown/XML artifacts that crash sandbox."""
        junk_patterns = ['```', '<output>', '</output>', '<answer>', '</answer>',
                         '<code>', '</code>', '<text>', '</text>']
        return any(p in code for p in junk_patterns)

    def send_bash_command(self, code, files=dict(), files_to_fetch=[]):
        # Validate command before sending
        if self.has_junk_artifacts(code):
            return {"status": "Failed", "run_result": {"stderr": "Command contains invalid artifacts"}}

        if not self.is_valid_bash_syntax(code):
            return {"status": "Failed", "run_result": {"stderr": "Invalid bash syntax"}}

        response = requests.post(self.url, json={
            'code': f'''{code}''',
            'language': 'bash',
            'run_timeout': 1,
            'files': files,
            'fetch_files': files_to_fetch,
        })

        return response.json()

    def decode_fetched_files(self, resp_json):
        import base64
        try:
            out_dict = dict()
            if  "files" not in resp_json:
                return np.array(dict())
            for k, v in resp_json["files"].items():
                out_dict[k] = base64.b64decode(v).decode('utf-8')
            # transform into numpy as DataProto expects arrays
            return np.array(out_dict)
        except Exception as e:
            print(f"Failed to decode file. {e=}")
            return np.array({})

    def create_command_output(self, result):
        if "status" not in result:
            print(f"status no in result. {result=}")
        if result.get("status", "") == "Success":
            return f"{result['run_result']['stdout']}"
        else:
            if "run_result" in result and "stderr" in result["run_result"]:
                return f"Execution Failed: {result['run_result']['stderr']}"
            else:
                print(f"\n\n\n\nExecution failed without std Err: {result=}\n\n\n\n")
                return f"Execution Failed: {result=}"

    def execute_agent_command(self, agent_command):
        """Execute a command from the agent with full history replay"""

        if self.command_history:
            # Replay entire history as a script
            state_script = "\n".join(self.command_history)


            # Put script in a file to avoid heredoc/quoting issues
            state_script_b64 = base64.b64encode(state_script.encode()).decode()

            files = self.files.copy()
            files['__replay_state.sh'] = state_script_b64

            full_command = f"""
source __replay_state.sh &> /dev/null
{agent_command}
"""
        else:
            # First command, no history
            files = self.files
            full_command = agent_command

        result = self.send_bash_command(full_command, files=files, files_to_fetch=self.files_to_fetch)

        # Add to history if execution succeeded
        if result.get('status') == "Success":
            self.command_history.append(agent_command)

        fetched_files = self.decode_fetched_files(result)

        return self.create_command_output(result), fetched_files

    async def run(
        self, sampling_params: dict[str, Any], *, cancellation_event: asyncio.Event = None, **kwargs
    ) -> AgentLoopOutput:
        check_server_running()

        assert "tools_kwargs" in kwargs
        import json
        self.tools_kwargs = json.loads(kwargs["tools_kwargs"])
        assert "files_dict" in self.tools_kwargs, f"{self.tools_kwargs=}"
        self.files_to_fetch = self.tools_kwargs.get("files_to_fetch", [])
        startup_commands = self.tools_kwargs.get("startup_commands", [])
        files_dict = self.tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"{files_dict=}"
        self.files = self.flatten_structure(files_dict)

        maybe_partial_output: Optional[AgentLoopOutput] = kwargs.get("output", None)
        param_version = kwargs.get("param_version", 0)
        param_version_start = param_version
        param_version_end = param_version
        request_id = uuid4().hex
        fetched_files = np.array(dict())
        max_num_turns = self.config.actor_rollout_ref.rollout.multi_turn.get("max_assistant_turns", 5)
        is_cancel = False

        if not maybe_partial_output:
            metrics = {}
            # empty command history for each run
            self.command_history = startup_commands
            num_turns = 0
            mask = list()
            messages = list(kwargs["raw_prompt"])
            # Track full conversation for logging
            conversation_messages = [dict(msg) for msg in messages]
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            # Commented as response mask shouldn't include the prompt
            # mask += [0] * len(prompt_ids)
            all_output_with_tool = list()
            curr_input = [tok for tok in prompt_ids]
            log_probs_all = list()
        else:
            if maybe_partial_output.extra_fields.get("is_cancel", False):
                metrics = maybe_partial_output.metrics
                param_version_start = maybe_partial_output.extra_fields["param_version_start"]
                self.command_history = maybe_partial_output.extra_fields["command_history"]
                conversation_messages = maybe_partial_output.extra_fields.get("messages", [])
                num_turns = maybe_partial_output.num_turns
                mask = list(maybe_partial_output.response_mask)
                prompt_ids = maybe_partial_output.prompt_ids
                all_output_with_tool = list(maybe_partial_output.response_ids)
                curr_input = (
                    maybe_partial_output.prompt_ids
                    + list(maybe_partial_output.response_ids)
                )
                log_probs_all = list(maybe_partial_output.response_logprobs) if maybe_partial_output.response_logprobs else list()
            else:
                # Already completed, return as-is
                return maybe_partial_output

        while num_turns < max_num_turns:
            # Check for external cancellation
            if cancellation_event and cancellation_event.is_set():
                is_cancel = True
                break

            # Use processor if available for multimodal support
            assert len(curr_input) > 0
            assert isinstance(curr_input, list)
            assert isinstance(curr_input[-1], int)

            token_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                request_id=request_id, prompt_ids=curr_input, sampling_params=sampling_params
            )

            assert isinstance(token_ids, list)
            if len(token_ids) == 0:
                # No tokens generated, treat as cancellation
                is_cancel = True
                break
            assert isinstance(token_ids[0], int)

            all_output_with_tool += token_ids
            mask += [1] * len(token_ids)
            if log_probs:
                log_probs_all += log_probs

            decoded_output = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(token_ids)
            )
            # Track assistant message for logging
            conversation_messages.append({
                "role": "assistant",
                "content": decoded_output
            })
            cmd = await self.loop.run_in_executor(
                None,
                lambda: self.extract_bash_command(decoded_output)
            )
            # if agent doesn't output a command, we exit the loop (completed)
            if cmd is None:
                break

            curr_input += token_ids

            # Check response length before tool execution
            if len(mask) >= self.response_length:
                break

            # Check for dangerous commands that could kill the environment
            dangerous_patterns = ["pkill", "kill ", "kill\t", "killall", "shutdown", "reboot", "halt", "poweroff", "rm -rf /", ":(){ :|:& };:"]
            is_dangerous = any(pattern in cmd for pattern in dangerous_patterns)

            if is_dangerous:
                # Return realistic permission error instead of executing
                cmd_output = f"bash: {cmd.split()[0]}: Operation not permitted"
                fetched_files = np.array(dict())
            else:
                cmd_output, fetched_files = await self.loop.run_in_executor(
                    None,
                    lambda: self.execute_agent_command(cmd)
                )
            cmd_message = [{
                "role": "tool",
                "content": cmd_output
            }]
            # Track tool message for logging
            conversation_messages.append({
                "role": "tool",
                "content": cmd_output
            })
            cmd_message_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    cmd_message, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            curr_input += cmd_message_ids
            all_output_with_tool += cmd_message_ids
            mask += [0] * len(cmd_message_ids)

            if len(mask) >= self.response_length:
                break

            # If cancelled, save state after executing any pending command
            if is_cancel:
                break

            num_turns += 1

        # Build output
        assert len(mask) == len(all_output_with_tool), f"{len(mask)=}, {len(all_output_with_tool)=}"

        # Truncate to response_length
        mask = mask[: self.response_length]
        all_output_with_tool = all_output_with_tool[: self.response_length]
        log_probs_all = log_probs_all[: self.response_length] if log_probs_all else None

        assert len(mask) == len(all_output_with_tool), f"{len(mask)=}, {len(all_output_with_tool)=}"

        assert isinstance(fetched_files, np.ndarray)
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:self.prompt_length],
            response_ids=all_output_with_tool,
            response_mask=mask,
            response_logprobs=log_probs_all,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=dict(
                fetched_files=fetched_files,
                command_history=self.command_history,
                messages=conversation_messages,
                is_cancel=is_cancel,
                param_version_start=param_version_start,
                param_version_end=param_version_end,
            ),
        )
        return output

