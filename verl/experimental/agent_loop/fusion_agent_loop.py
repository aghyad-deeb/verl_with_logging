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
import logging
import os
import base64
import numpy as np
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.workers.rollout.replica import TokenOutput
from verl.utils.profiler import simple_timer

# Official SandboxFusion SDK - provides built-in retries, async support, proper error handling
from sandbox_fusion import (
    run_code_async,
    run_code,
    RunCodeRequest,
    RunCodeResponse,
    RunStatus,
    set_endpoint,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration
SANDBOX_ENDPOINT = os.getenv("SANDBOX_FUSION_ENDPOINT", "http://localhost:60808")
SANDBOX_MAX_RETRIES = int(os.getenv("SANDBOX_MAX_RETRIES", "5"))
SANDBOX_CLIENT_TIMEOUT = float(os.getenv("SANDBOX_CLIENT_TIMEOUT", "5"))
SANDBOX_RUN_TIMEOUT = float(os.getenv("SANDBOX_RUN_TIMEOUT", "1"))
SANDBOX_HEALTH_CHECK_MAX_RETRIES = int(os.getenv("SANDBOX_HEALTH_CHECK_MAX_RETRIES", "5"))

# Track if health check has been done (class-level, once per process)
_health_check_done = False

def check_server_running(force: bool = False) -> bool:
    """Check if sandbox server is running using the official SDK with exponential backoff.
    
    This is a one-time check per process to avoid thundering herd when
    many workers initialize simultaneously in distributed training.
    
    Args:
        force: If True, always perform the check even if already done.
        
    Returns:
        True if server is running, raises RuntimeError otherwise.
    """
    global _health_check_done
    
    # Skip if already checked (unless forced)
    if _health_check_done and not force:
        return True
    
    try:
        result = run_code(
            RunCodeRequest(code='echo "health_check"', language='bash', run_timeout=2),
            endpoint=SANDBOX_ENDPOINT,
            max_attempts=SANDBOX_HEALTH_CHECK_MAX_RETRIES,
            client_timeout=5
        )
        _health_check_done = True
        return result.status == RunStatus.Success
    except Exception as e:
        raise RuntimeError(
            f"Sandbox server is not running at '{SANDBOX_ENDPOINT}'. "
            f"Start it with: docker run -it -p 60808:8080 volcengine/sandbox-fusion:server-20250609"
        ) from e


@register("fusion_agent_loop")
class FusionAgentLoop(AgentLoopBase):
    """
    Agent loop with bash sandbox execution using the official SandboxFusion SDK.
    
    Features:
    - Built-in retry logic (configurable via SANDBOX_MAX_RETRIES env var)
    - Native async support for better performance
    - Proper timeout handling (configurable via SANDBOX_CLIENT_TIMEOUT, SANDBOX_RUN_TIMEOUT)
    - Statefulness via command history replay
    
    Environment variables:
    - SANDBOX_FUSION_ENDPOINT: Sandbox server URL (default: http://localhost:60808)
    - SANDBOX_MAX_RETRIES: Max retry attempts for failed requests (default: 5)
    - SANDBOX_CLIENT_TIMEOUT: HTTP client timeout in seconds (default: 30)
    - SANDBOX_RUN_TIMEOUT: Code execution timeout in seconds (default: 10)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # SDK configuration
        self.endpoint = SANDBOX_ENDPOINT
        self.max_retries = SANDBOX_MAX_RETRIES
        self.client_timeout = SANDBOX_CLIENT_TIMEOUT
        self.run_timeout = SANDBOX_RUN_TIMEOUT
        
        # Set global endpoint for SDK
        set_endpoint(self.endpoint)
        

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

    async def is_valid_bash_syntax(self, code: str) -> bool:
        """Check if command has valid bash syntax using bash -n in sandbox.
        
        Uses the SDK's async interface with minimal retries for validation.
        """
        script_b64 = base64.b64encode(code.encode()).decode()
        try:
            result = await run_code_async(
                RunCodeRequest(
                    code='bash -n __validate_cmd.sh',
                    language='bash',
                    run_timeout=2,
                    files={'__validate_cmd.sh': script_b64},
                ),
                endpoint=self.endpoint,
                max_attempts=5,  # Fewer retries for validation
                client_timeout=5
            )
            # Check both top-level status and run_result return_code
            if result.status != RunStatus.Success:
                return False
            return result.run_result is not None and result.run_result.return_code == 0
        except Exception as e:
            logger.debug(f"Bash syntax validation failed: {e}")
            return False

    def has_junk_artifacts(self, code):
        """Check for markdown/XML artifacts that crash sandbox."""
        junk_patterns = ['```', '<output>', '</output>', '<answer>', '</answer>',
                         '<code>', '</code>', '<text>', '</text>']
        return any(p in code for p in junk_patterns)

    async def send_bash_command(
        self, 
        code: str, 
        files: Optional[dict] = None, 
        files_to_fetch: Optional[list] = None,
        skip_validation: bool = False
    ) -> dict:
        """Execute bash command using the official SandboxFusion SDK.
        
        Args:
            code: The bash code to execute
            files: Dict of filename -> base64-encoded content
            files_to_fetch: List of file paths to retrieve after execution
            skip_validation: Skip syntax validation (for pre-validated commands)
            
        Returns:
            Dict with status, run_result, and files
        """
        files = files or {}
        files_to_fetch = files_to_fetch or []
        
        # Validate command before sending
        if self.has_junk_artifacts(code):
            return {"status": "Failed", "run_result": {"stderr": "Command contains invalid artifacts"}}

        if not skip_validation and not await self.is_valid_bash_syntax(code):
            return {"status": "Failed", "run_result": {"stderr": "Invalid bash syntax"}}

        try:
            result = await run_code_async(
                RunCodeRequest(
                    code=code,
                    language='bash',
                    run_timeout=self.run_timeout,
                    files=files,
                    fetch_files=files_to_fetch,
                ),
                endpoint=self.endpoint,
                max_attempts=self.max_retries,
                client_timeout=self.client_timeout
            )
            
            # Convert SDK response to dict format for backward compatibility
            return {
                "status": result.status.value,  # "Success", "Failed", or "SandboxError"
                "message": result.message,
                "run_result": {
                    "status": result.run_result.status.value if result.run_result else None,
                    "stdout": result.run_result.stdout if result.run_result else "",
                    "stderr": result.run_result.stderr if result.run_result else "",
                    "return_code": result.run_result.return_code if result.run_result else None,
                    "execution_time": result.run_result.execution_time if result.run_result else None,
                } if result.run_result else {},
                "files": result.files or {},
            }
        except Exception as e:
            logger.error(f"Sandbox execution failed after {self.max_retries} retries: {e}")
            return {
                "status": "Failed",
                "run_result": {"stderr": f"Sandbox error: {str(e)}"},
                "files": {}
            }

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
            if "run_result" in result and "stderr" in result.get("run_result", ""):
                return f"Execution Failed: {result['run_result']['stderr']}"
            else:
                print(f"\n\n\n\nExecution failed without std Err: {result=}\n\n\n\n")
                return f"Execution Failed: {result=}"

    async def execute_agent_command(self, agent_command: str) -> tuple:
        """Execute a command from the agent with full history replay for statefulness.
        
        This implements statefulness by replaying the entire command history before
        executing the new command. This ensures that:
        - Directory changes (cd) persist
        - Environment variables (export) persist
        - Shell variables persist
        - File system changes persist
        
        Args:
            agent_command: The bash command to execute
            
        Returns:
            Tuple of (command_output_string, fetched_files_array)
        """
        if self.command_history:
            # Replay entire history as a script
            state_script = "\n".join(self.command_history)

            # Put script in a file to avoid heredoc/quoting issues
            state_script_b64 = base64.b64encode(state_script.encode()).decode()

            files = self.files.copy()
            files['__replay_state.sh'] = state_script_b64

            full_command = f"""source __replay_state.sh &> /dev/null
{agent_command}"""
        else:
            # First command, no history
            files = self.files
            full_command = agent_command

        # Use async SDK call directly (no run_in_executor needed)
        result = await self.send_bash_command(
            full_command, 
            files=files, 
            files_to_fetch=self.files_to_fetch,
            skip_validation=False  # Still validate the full command
        )

        # Add to history if execution succeeded
        if result.get('status') == "Success":
            self.command_history.append(agent_command)

        fetched_files = self.decode_fetched_files(result)

        return self.create_command_output(result), fetched_files

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # NOTE: Health check removed - SDK retries handle failures gracefully

        assert "tools_kwargs" in kwargs
        import json
        self.tools_kwargs=json.loads(kwargs["tools_kwargs"])
        assert  "files_dict" in self.tools_kwargs, f"{self.tools_kwargs=}"
        self.files_to_fetch = self.tools_kwargs.get("files_to_fetch", [])
        startup_commands = self.tools_kwargs.get("startup_commands", [])
        files_dict = self.tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"{files_dict=}"
        self.files = self.flatten_structure(files_dict)

        messages = list(kwargs["raw_prompt"])
        # Track full conversation for logging
        conversation_messages = [dict(msg) for msg in messages]
        metrics = {}
        # empty command history for each run
        self.command_history = startup_commands
        request_id = uuid4().hex
        num_turns = 0
        max_num_turns = self.config.actor_rollout_ref.rollout.multi_turn.get("max_assistant_turns", 5)
        mask = list()
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        # Commented as response mask shouldn't include the prompt
        # mask += [0] * len(prompt_ids)
        curr_input = [tok for tok in prompt_ids]
        all_output_with_tool = list()
        import numpy as np
        fetched_files = np.array(dict())
        with simple_timer("generate_sequences_all_turns", metrics):
            while num_turns < max_num_turns:
                # Use processor if available for multimodal support
                assert len(curr_input) > 0
                assert isinstance(curr_input, list)
                assert isinstance(curr_input[-1], int)

                # if len(curr_input) >
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=curr_input, sampling_params=sampling_params
                )
                #! This will fail but we'll get to know the type
                assert isinstance(output, TokenOutput), f"{type(output)=}, {output=}"
                #! Edit once figured out
                # output_tokens = [output.token_ids
                # TODO next: figure out problem with wandb api key,find the type of output,

                assert isinstance(output.token_ids, list)
                assert isinstance(output.token_ids[0], int) #! will fail if len is 0, but shouldn't ever be
                all_output_with_tool += output.token_ids
                mask += [1] * len(output.token_ids)
                decoded_output = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(output.token_ids)
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

                # if agent doesn't output a command, we exit the loop
                if cmd is None:
                    break

                curr_input += output.token_ids

                # Check for dangerous commands that could kill the environment
                dangerous_patterns = [
                    # Process/system killing
                    "pkill", "kill ", "kill\t", "killall", "shutdown", "reboot", "halt", "poweroff",
                    # Destructive
                    "rm -rf /", ":(){ :|:& };:",
                    # Privilege escalation
                    "sudo",
                    # Infinite loops (crash sandbox)
                    "while true", "while :", "for ((",
                    # Arbitrary code execution
                    "exec(", "eval(",
                    # Malicious intent
                    "malware", "exploit", "payload",
                    # Credential/system access
                    "contact_server", "/var/run/",
                    # Destructive scripts
                    "destroy",
                    # Network scanning/recon
                    "nmap", "netcat", " nc ",
                    # Piped shell execution
                    "| sh", "| bash",
                    # Hidden directory execution
                    "/.config/", "/.local/",
                    # Temp directory execution
                    "/tmp/mutation/",
                    # Shell spawning
                    "$SHELL",
                    # Binary compilation (hide malicious code)
                    "nuitka",
                    # External IP requests (data exfiltration)
                    "curl -X POST http://",
                    # Undefined suspicious functions
                    "readd_bash_add",
                    # Interactive input (HANGS sandbox waiting for user)
                    "read -p",
                    # Network commands
                    "ping ", "ping\t",
                    # Root filesystem search (slow/heavy)
                    "find /",
                    # Device reads (slow/hang)
                    "/dev/urandom", "/dev/random", "/dev/zero",
                    # Malformed input that crashes sandbox
                    "```",
                    # Blocking commands
                    "tail -f", "watch ",
                ]
                is_dangerous = any(pattern in cmd for pattern in dangerous_patterns)

                if is_dangerous:
                    # Return realistic permission error instead of executing
                    cmd_output = f"bash: {cmd.split()[0]}: Operation not permitted"
                    fetched_files = np.array(dict())
                else:
                    # Use native async - no run_in_executor needed with SDK
                    cmd_output, fetched_files = await self.execute_agent_command(cmd)
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

                num_turns += 1

        # response_mask = [1] * len(output.token_ids)
        assert len(mask) == len(all_output_with_tool), f"{len(mask)=}, {len(all_output_with_tool)=}, {mask=}\n{all_output_with_tool=}"
        mask = mask[: self.response_length]
        all_output_with_tool = all_output_with_tool[: self.response_length]
        assert len(mask) == len(all_output_with_tool), f"{len(mask)=}, {len(all_output_with_tool)=}, {mask=}\n{all_output_with_tool=}"

        assert isinstance(fetched_files, np.ndarray)
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:self.prompt_length],
            # prompt_ids=,
            response_ids=all_output_with_tool[: self.response_length],
            # response_ids=, #! I don't think I want these here
            response_mask=mask[: self.response_length],
            # response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            # response_logprobs=,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=dict(
                fetched_files=fetched_files,
                messages=conversation_messages,
            )
        )
        return output
