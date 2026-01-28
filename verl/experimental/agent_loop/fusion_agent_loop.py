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
Fusion Agent Loop - Stateful Bash Execution via SandboxFusion Sessions.

This is the main agent loop for bash execution with true statefulness:
- Working directory changes (cd) persist across commands
- Environment variables (export) persist across commands
- File system changes persist within the episode
- Full isolation between episodes (session cleanup)

Uses SandboxFusion's session API which maintains state via subprocess pipes,
allowing for 10,000+ concurrent sessions.

Requirements:
    - SandboxFusion server with session support running
    - See: https://github.com/aghyad-deeb/sandbox-fusion-sessions
    - Session API endpoints: /session/create, /session/run, /session/destroy

Environment variables:
    - SANDBOX_FUSION_ENDPOINT: Sandbox server URL (default: http://localhost:60808)
    - SANDBOX_CLIENT_TIMEOUT: HTTP client timeout in seconds (default: 120)
    - SANDBOX_RUN_TIMEOUT: Code execution timeout in seconds (default: 10)
"""
import logging
import os
import base64
import numpy as np
import aiohttp
from typing import Any, Optional, Dict, List
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.workers.rollout.replica import TokenOutput
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration
# Support multiple endpoints for load distribution via consistent hashing
# Format: comma-separated URLs, e.g., "http://localhost:60808,http://localhost:60809"
SANDBOX_ENDPOINTS_STR = os.getenv(
    "SANDBOX_FUSION_ENDPOINTS",
    os.getenv("SANDBOX_FUSION_ENDPOINT", "http://localhost:60808")
)
SANDBOX_ENDPOINTS = [e.strip() for e in SANDBOX_ENDPOINTS_STR.split(",") if e.strip()]

# Legacy single endpoint for backward compatibility
SANDBOX_ENDPOINT = SANDBOX_ENDPOINTS[0] if SANDBOX_ENDPOINTS else "http://localhost:60808"


def check_server_running(url: str = None) -> bool:
    """Check if the SandboxFusion session server(s) are running.
    
    Args:
        url: Server URL to check. If None, checks all configured endpoints.
        
    Returns:
        True if server(s) are healthy, False otherwise.
    """
    import requests
    
    endpoints_to_check = [url] if url else SANDBOX_ENDPOINTS
    
    for endpoint in endpoints_to_check:
        try:
            resp = requests.get(f"{endpoint}/v1/ping", timeout=5)
            if resp.status_code != 200 or "pong" not in resp.text:
                logger.debug(f"Server check failed for {endpoint}")
                return False
        except Exception as e:
            logger.debug(f"Server check failed for {endpoint}: {e}")
            return False
    
    return True


SANDBOX_CLIENT_TIMEOUT = float(os.getenv("SANDBOX_CLIENT_TIMEOUT", "120"))
SANDBOX_RUN_TIMEOUT = float(os.getenv("SANDBOX_RUN_TIMEOUT", "10"))


class SessionClient:
    """
    Async client for SandboxFusion session-based bash execution.
    
    Supports multiple container endpoints with consistent hashing for load distribution.
    Sessions are routed to containers based on a hash of the session_id, ensuring
    all requests for a session go to the same container.
    
    Provides high-level methods for creating, using, and destroying
    stateful bash sessions.
    """
    
    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        client_timeout: float = SANDBOX_CLIENT_TIMEOUT,
        run_timeout: float = SANDBOX_RUN_TIMEOUT,
    ):
        """
        Initialize the session client.
        
        Args:
            endpoints: List of container endpoints. If None, uses SANDBOX_ENDPOINTS.
            client_timeout: HTTP client timeout in seconds.
            run_timeout: Default command execution timeout in seconds.
        """
        import hashlib
        self._hashlib = hashlib
        
        # Support both single endpoint (backward compat) and multiple endpoints
        if endpoints is None:
            endpoints = SANDBOX_ENDPOINTS
        elif isinstance(endpoints, str):
            endpoints = [endpoints]
        
        self.endpoints = [e.rstrip('/') for e in endpoints]
        self.client_timeout = aiohttp.ClientTimeout(total=client_timeout)
        self.run_timeout = run_timeout
        
        # One HTTP session per endpoint for connection pooling
        self._http_sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # Track session_id -> endpoint mapping for consistent routing
        self._session_endpoints: Dict[str, str] = {}
    
    def _hash_to_endpoint(self, session_id: str) -> str:
        """
        Consistent hash: same session_id always maps to same endpoint.
        Uses MD5 for stability across Python versions (built-in hash() can vary).
        """
        if len(self.endpoints) == 1:
            return self.endpoints[0]
        
        hash_bytes = self._hashlib.md5(session_id.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], 'big')
        return self.endpoints[hash_int % len(self.endpoints)]
    
    def _get_endpoint_for_session(self, session_id: str) -> str:
        """Get the endpoint for a session, using cached mapping or hash."""
        if session_id in self._session_endpoints:
            return self._session_endpoints[session_id]
        
        endpoint = self._hash_to_endpoint(session_id)
        self._session_endpoints[session_id] = endpoint
        return endpoint
    
    async def _get_http_session(self, endpoint: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for an endpoint (connection pooling)."""
        if endpoint not in self._http_sessions or self._http_sessions[endpoint].closed:
            self._http_sessions[endpoint] = aiohttp.ClientSession(timeout=self.client_timeout)
        return self._http_sessions[endpoint]
    
    async def close(self):
        """Close all HTTP sessions."""
        for session in self._http_sessions.values():
            if session and not session.closed:
                await session.close()
        self._http_sessions.clear()
        self._session_endpoints.clear()
    
    async def create_session(
        self,
        session_id: str,
        files: Optional[Dict[str, str]] = None,
        startup_commands: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new bash session.
        
        The session_id is generated client-side and used for consistent hashing
        to determine which container handles this session.
        
        Args:
            session_id: Client-generated session identifier (used for routing)
            files: Dict of filename -> base64-encoded content
            startup_commands: Commands to run on initialization
            env: Additional environment variables
            
        Returns:
            session_id for use in subsequent commands
        """
        endpoint = self._get_endpoint_for_session(session_id)
        http_session = await self._get_http_session(endpoint)
        
        payload = {
            "session_id": session_id,  # Pass client-generated ID to server
            "files": files or {},
            "startup_commands": startup_commands or [],
            "env": env or {},
        }
        
        async with http_session.post(
            f"{endpoint}/session/create",
            json=payload,
        ) as response:
            result = await response.json()
            
        if result.get("status") != "Success":
            raise RuntimeError(f"Failed to create session: {result.get('message', 'Unknown error')}")
        
        return session_id
    
    async def run_command(
        self,
        session_id: str,
        command: str,
        timeout: Optional[float] = None,
        fetch_files: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run a command in an existing session.
        
        Automatically routes to the correct container based on session_id hash.
        
        Args:
            session_id: Session identifier
            command: Bash command to execute
            timeout: Command timeout (uses default if not specified)
            fetch_files: Files to retrieve after execution
            
        Returns:
            Dict with status, stdout, stderr, return_code, files
        """
        endpoint = self._get_endpoint_for_session(session_id)
        http_session = await self._get_http_session(endpoint)
        
        payload = {
            "session_id": session_id,
            "command": command,
            "timeout": timeout or self.run_timeout,
            "fetch_files": fetch_files or [],
        }
        
        async with http_session.post(
            f"{endpoint}/session/run",
            json=payload,
        ) as response:
            return await response.json()
    
    async def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session and clean up resources.
        
        Automatically routes to the correct container based on session_id hash.
        
        Args:
            session_id: Session to destroy
            
        Returns:
            True if successful
        """
        endpoint = self._get_endpoint_for_session(session_id)
        http_session = await self._get_http_session(endpoint)
        
        payload = {"session_id": session_id}
        
        try:
            async with http_session.post(
                f"{endpoint}/session/destroy",
                json=payload,
            ) as response:
                result = await response.json()
                # Clean up the endpoint mapping
                self._session_endpoints.pop(session_id, None)
                return result.get("status") == "Success"
        except Exception as e:
            logger.warning(f"Failed to destroy session {session_id}: {e}")
            self._session_endpoints.pop(session_id, None)
            return False


@register("fusion_agent_loop")
class FusionAgentLoop(AgentLoopBase):
    """
    Stateful agent loop using SandboxFusion session-based bash execution.
    
    This provides true bash statefulness:
    - cd, export, file changes persist across commands
    - No command replay overhead
    - Full isolation between episodes
    
    Features:
    - High performance: persistent bash process, no replay
    - True statefulness: shell variables, functions, aliases persist
    - Episode isolation: session cleanup removes all state
    - Concurrent support: each episode gets its own session
    
    Environment variables:
    - SANDBOX_FUSION_ENDPOINT: Server URL (default: http://localhost:60808)
    - SANDBOX_CLIENT_TIMEOUT: HTTP timeout (default: 120)
    - SANDBOX_RUN_TIMEOUT: Execution timeout (default: 10)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Session client for stateful bash (supports multiple containers)
        self.session_client = SessionClient(
            endpoints=SANDBOX_ENDPOINTS,
            client_timeout=SANDBOX_CLIENT_TIMEOUT,
            run_timeout=SANDBOX_RUN_TIMEOUT,
        )
        
        # Current session ID (set per episode)
        self.current_session_id: Optional[str] = None

    def flatten_structure(self, fs_list, prefix=""):
        """Convert nested file structure to flat dict with base64 content."""
        files = {}
        for item in fs_list:
            path = f"{prefix}/{item['name']}" if prefix else item['name']
            if item['type'] == 'file':
                files[path] = base64.b64encode(item['content'].encode()).decode()
            else:
                files.update(self.flatten_structure(item['content'], path))
        return files

    def extract_bash_command(self, text, prefix="<bash>", suffix="</bash>"):
        """Extract bash command from model output."""
        assert isinstance(text, str), f"text must be a string, got {type(text)}"
        assert isinstance(prefix, str), f"prefix must be a string, got {type(prefix)}"
        assert isinstance(suffix, str), f"suffix must be a string, got {type(suffix)}"
        assert len(prefix) > 0, "prefix cannot be empty"
        assert len(suffix) > 0, "suffix cannot be empty"
        
        eot = "</think>"
        if eot in text:
            text = text.split(eot)[-1]
            
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

    def has_junk_artifacts(self, code):
        """Check for markdown/XML artifacts that could cause issues."""
        junk_patterns = ['```', '<output>', '</output>', '<answer>', '</answer>',
                         '<code>', '</code>', '<text>', '</text>']
        return any(p in code for p in junk_patterns)

    def decode_fetched_files(self, files_dict: Dict[str, str]) -> np.ndarray:
        """Decode base64 files to string content."""
        try:
            out_dict = {}
            for k, v in files_dict.items():
                out_dict[k] = base64.b64decode(v).decode('utf-8')
            return np.array(out_dict)
        except Exception as e:
            logger.warning(f"Failed to decode files: {e}")
            return np.array({})

    def create_command_output(self, result: Dict) -> str:
        """Format command result for the model."""
        if result.get("status") == "Success":
            return result.get("stdout", "")
        else:
            stderr = result.get("stderr", "")
            message = result.get("message", "")
            if stderr:
                return f"Execution Failed: {stderr}"
            elif message:
                return f"Execution Failed: {message}"
            else:
                return f"Execution Failed: {result}"

    async def execute_agent_command(self, command: str) -> tuple:
        """
        Execute a command in the current session.
        
        The session maintains all state, so no history replay is needed.
        
        Args:
            command: Bash command to execute
            
        Returns:
            Tuple of (command_output_string, fetched_files_array)
        """
        assert self.current_session_id is not None, "Session not initialized"
        
        # Ensure command ends with newline - required for heredocs where the
        # delimiter must be on its own line (e.g., EOF must not be followed by </bash>)
        if not command.endswith('\n'):
            command = command + '\n'
        
        result = await self.session_client.run_command(
            session_id=self.current_session_id,
            command=command,
            timeout=self.session_client.run_timeout,
            fetch_files=self.files_to_fetch,
        )
        
        fetched_files = self.decode_fetched_files(result.get("files", {}))
        return self.create_command_output(result), fetched_files

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one episode of the agent loop with a stateful session."""
        
        # Parse tools kwargs
        assert "tools_kwargs" in kwargs
        import json
        self.tools_kwargs = json.loads(kwargs["tools_kwargs"])
        assert "files_dict" in self.tools_kwargs, f"{self.tools_kwargs=}"
        self.files_to_fetch = self.tools_kwargs.get("files_to_fetch", [])
        startup_commands = self.tools_kwargs.get("startup_commands", [])
        files_dict = self.tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"{files_dict=}"
        files = self.flatten_structure(files_dict)

        # Create a new session for this episode
        # Generate session_id client-side for consistent hashing across multiple containers
        session_id = uuid4().hex
        metrics = {}
        
        with simple_timer("session_create", metrics):
            try:
                self.current_session_id = await self.session_client.create_session(
                    session_id=session_id,
                    files=files,
                    startup_commands=startup_commands,
                )
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise

        try:
            messages = list(kwargs["raw_prompt"])
            conversation_messages = [dict(msg) for msg in messages]
            metrics["tool_calls_count"] = 0
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
            
            curr_input = [tok for tok in prompt_ids]
            all_output_with_tool = list()
            fetched_files = np.array(dict())
            
            with simple_timer("generate_sequences_all_turns", metrics):
                while num_turns < max_num_turns:
                    assert len(curr_input) > 0
                    assert isinstance(curr_input, list)
                    assert isinstance(curr_input[-1], int)

                    with simple_timer("generate_sequences", metrics):
                        output = await self.server_manager.generate(
                            request_id=request_id, prompt_ids=curr_input, sampling_params=sampling_params
                        )
                    assert isinstance(output, TokenOutput), f"{type(output)=}, {output=}"
                    assert isinstance(output.token_ids, list)
                    assert isinstance(output.token_ids[0], int)
                    
                    all_output_with_tool += output.token_ids
                    mask += [1] * len(output.token_ids)
                    
                    decoded_output = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.decode(output.token_ids)
                    )
                    
                    conversation_messages.append({
                        "role": "assistant",
                        "content": decoded_output
                    })
                    
                    cmd = await self.loop.run_in_executor(
                        None,
                        lambda: self.extract_bash_command(decoded_output)
                    )

                    # Exit if no command
                    if cmd is None:
                        break

                    curr_input += output.token_ids

                    # Check for dangerous commands
                    dangerous_patterns = [
                        "pkill", "kill ", "kill\t", "killall", "shutdown", "reboot", "halt", "poweroff",
                        "rm -rf /", ":(){ :|:& };:",
                        "sudo",
                        "while true", "while :", "for ((",
                        "exec(", "eval(",
                        "malware", "exploit", "payload",
                        "contact_server", "/var/run/",
                        "destroy",
                        "nmap", "netcat", " nc ",
                        "| sh", "| bash",
                        "/.config/", "/.local/",
                        "/tmp/mutation/",
                        "$SHELL",
                        "nuitka",
                        "curl -X POST http://",
                        "readd_bash_add",
                        "read -p",
                        "ping ", "ping\t",
                        "find /",
                        "/dev/urandom", "/dev/random", "/dev/zero",
                        "```",
                        "tail -f", "watch ",
                    ]
                    is_dangerous = any(pattern in cmd for pattern in dangerous_patterns)

                    if is_dangerous:
                        cmd_output = f"bash: {cmd.split()[0]}: Operation not permitted"
                        fetched_files = np.array(dict())
                    elif self.has_junk_artifacts(cmd):
                        cmd_output = "Command contains invalid artifacts"
                        fetched_files = np.array(dict())
                    else:
                        with simple_timer("tool_calls", metrics):
                            cmd_output, fetched_files = await self.execute_agent_command(cmd)
                        metrics["tool_calls_count"] = metrics.get("tool_calls_count", 0) + 1
                    
                    cmd_message = [{
                        "role": "tool",
                        "content": cmd_output
                    }]
                    
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

            # Prepare output
            assert len(mask) == len(all_output_with_tool)
            mask = mask[:self.response_length]
            all_output_with_tool = all_output_with_tool[:self.response_length]
            assert len(mask) == len(all_output_with_tool)

            # Final file fetch - ensure we get the requested files even if the last
            # command was blocked or had issues
            if self.files_to_fetch and self.current_session_id:
                with simple_timer("final_file_fetch", metrics):
                    try:
                        final_result = await self.session_client.run_command(
                            session_id=self.current_session_id,
                            command="true",  # No-op command just to trigger file fetch
                            timeout=5,
                            fetch_files=self.files_to_fetch,
                        )
                        fetched_files = self.decode_fetched_files(final_result.get("files", {}))
                    except Exception as e:
                        logger.warning(f"Final file fetch failed: {e}")

            assert isinstance(fetched_files, np.ndarray)
            
            return AgentLoopOutput(
                prompt_ids=prompt_ids[:self.prompt_length],
                response_ids=all_output_with_tool[:self.response_length],
                response_mask=mask[:self.response_length],
                response_logprobs=output.log_probs[:self.response_length] if output.log_probs else None,
                num_turns=num_turns,
                metrics=metrics,
                extra_fields=dict(
                    fetched_files=fetched_files,
                    messages=conversation_messages,
                )
            )
            
        finally:
            # Always clean up the session at episode end
            if self.current_session_id:
                with simple_timer("session_destroy", metrics):
                    await self.session_client.destroy_session(self.current_session_id)
                self.current_session_id = None
            # Close the HTTP client sessions to avoid "Unclosed client session" warnings
            # Each episode creates its own FusionAgentLoop instance via hydra.utils.instantiate
            if self.session_client:
                with simple_timer("session_close_http", metrics):
                    await self.session_client.close()