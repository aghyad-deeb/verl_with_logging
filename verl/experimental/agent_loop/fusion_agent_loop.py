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
Fusion Agent Loop - Stateful bash execution via centralized session server.

This agent loop communicates with a SweRex Session Server via HTTP to execute
bash commands with true statefulness (cd, exports, variables persist).

Key benefits:
- True statefulness (no command replay needed)
- No local pool management
- No Ray worker detection needed  
- Single point of configuration (server URL)
- Server handles load balancing, health checks, fault tolerance

Environment variables:
    SWEREX_SERVER_URL: Session server URL (default: http://localhost:8180)
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import aiohttp
import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.workers.rollout.replica import TokenOutput
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def check_server_running(url: str = None) -> bool:
    """Check if the SweRex session server is running.
    
    Args:
        url: Server URL to check. Defaults to SWEREX_SERVER_URL env var.
        
    Returns:
        True if server is healthy, False otherwise.
    """
    import requests
    
    server_url = url or os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("status") == "healthy"
        return False
    except Exception as e:
        logger.debug(f"Server check failed: {e}")
        return False


# =============================================================================
# Configuration
# =============================================================================

SWEREX_SERVER_URL = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
SWEREX_REQUEST_TIMEOUT = float(os.getenv("SWEREX_REQUEST_TIMEOUT", "120"))
SWEREX_COMMAND_TIMEOUT = float(os.getenv("SWEREX_COMMAND_TIMEOUT", "30"))
# Client-side rate limiting: max concurrent acquire requests per process
# This prevents overwhelming the server when many agent loops run in parallel
SWEREX_MAX_CONCURRENT_ACQUIRE = int(os.getenv("SWEREX_MAX_CONCURRENT_ACQUIRE", "32"))

# =============================================================================
# HTTP Client
# =============================================================================

# Module-level client session (reused across requests)
_client_session: Optional[aiohttp.ClientSession] = None
# Module-level semaphore for rate-limiting acquire requests
_acquire_semaphore: Optional[asyncio.Semaphore] = None


def _get_acquire_semaphore() -> asyncio.Semaphore:
    """Get or create the shared acquire semaphore for rate limiting."""
    global _acquire_semaphore
    if _acquire_semaphore is None:
        _acquire_semaphore = asyncio.Semaphore(SWEREX_MAX_CONCURRENT_ACQUIRE)
        logger.info(f"Initialized client-side acquire rate limiter: max {SWEREX_MAX_CONCURRENT_ACQUIRE} concurrent")
    return _acquire_semaphore


async def get_client() -> aiohttp.ClientSession:
    """Get or create the shared HTTP client session."""
    global _client_session
    if _client_session is None or _client_session.closed:
        _client_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=SWEREX_REQUEST_TIMEOUT),
            # Limit concurrent connections to match our acquire rate limit
            connector=aiohttp.TCPConnector(limit=SWEREX_MAX_CONCURRENT_ACQUIRE * 2),
        )
    return _client_session


@dataclass
class CommandResult:
    """Result from command execution."""
    status: str
    stdout: str
    stderr: str
    return_code: int
    files: Optional[dict[str, str]] = None  # filename -> base64 content


# =============================================================================
# HTTP Agent Loop
# =============================================================================


@register("fusion_agent_loop")
class FusionAgentLoop(AgentLoopBase):
    """
    Agent loop with stateful bash execution via centralized session server.
    
    This agent loop communicates with a SweRex Session Server via HTTP.
    The server manages the pool of containers and sessions, handles load
    balancing, health checks, and fault tolerance.
    
    Key features:
    - True statefulness (cd, exports, variables persist across commands)
    - No local pool management (server handles everything)
    - No Ray worker detection (server handles load balancing)
    - Simple HTTP calls to acquire/execute/release
    
    Environment variables:
        SWEREX_SERVER_URL: Session server URL (default: http://localhost:8180)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Server configuration
        self.server_url = SWEREX_SERVER_URL
        
        # Session state (set during run())
        self.session_id: Optional[str] = None
        
        # Files and configuration (set during run())
        self.files: dict = {}
        self.files_to_fetch: list = []
    
    # =========================================================================
    # HTTP Operations
    # =========================================================================
    
    async def _acquire_session(
        self,
        files: Optional[dict[str, str]] = None,
        startup_commands: Optional[list[str]] = None,
    ) -> str:
        """Acquire a session from the server.
        
        Uses client-side rate limiting to prevent overwhelming the server
        with too many concurrent acquire requests.
        """
        # #region agent log
        import time as _time; import os as _os; _log_path = "./tmp/fusion_debug.log"; _os.makedirs(_os.path.dirname(_log_path), exist_ok=True); _acq_start = _time.time(); _payload_size = len(json.dumps({"files": files} if files else {})); open(_log_path, "a").write(json.dumps({"hypothesisId": "A", "location": "fusion_agent_loop.py:_acquire_session:start", "message": "acquire_session_start", "data": {"payload_size_bytes": _payload_size, "num_files": len(files) if files else 0, "server_url": self.server_url}, "timestamp": int(_time.time()*1000)}) + "\n")
        # #endregion
        
        # Rate limit: acquire semaphore before making HTTP request
        semaphore = _get_acquire_semaphore()
        # #region agent log
        _sem_start = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "F_client_ratelimit", "location": "fusion_agent_loop.py:_acquire_session:sem_wait", "message": "waiting_for_client_semaphore", "data": {}, "timestamp": int(_time.time()*1000)}) + "\n")
        # #endregion
        async with semaphore:
            # #region agent log
            _sem_acquired = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "F_client_ratelimit", "location": "fusion_agent_loop.py:_acquire_session:sem_acquired", "message": "client_semaphore_acquired", "data": {"semaphore_wait_ms": int((_sem_acquired - _sem_start)*1000)}, "timestamp": int(_time.time()*1000)}) + "\n")
            # #endregion
            
            client = await get_client()
            
            payload = {}
            if files:
                payload["files"] = files
            if startup_commands:
                payload["startup_commands"] = startup_commands
            
            # #region agent log
            _http_start = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "B", "location": "fusion_agent_loop.py:_acquire_session:http_start", "message": "http_post_starting", "data": {"time_to_get_client_ms": int((_http_start - _acq_start)*1000)}, "timestamp": int(_time.time()*1000)}) + "\n")
            # #endregion
            async with client.post(
                f"{self.server_url}/session/acquire",
                json=payload,
            ) as resp:
                # #region agent log
                _http_end = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "A", "location": "fusion_agent_loop.py:_acquire_session:http_done", "message": "http_post_completed", "data": {"http_duration_ms": int((_http_end - _http_start)*1000), "status": resp.status}, "timestamp": int(_time.time()*1000)}) + "\n")
                # #endregion
                if resp.status != 200:
                    error = await resp.text()
                    # #region agent log
                    open(_log_path, "a").write(json.dumps({"hypothesisId": "C", "location": "fusion_agent_loop.py:_acquire_session:error", "message": "acquire_failed", "data": {"status": resp.status, "error": error[:500]}, "timestamp": int(_time.time()*1000)}) + "\n")
                    # #endregion
                    raise RuntimeError(f"Failed to acquire session: {resp.status} - {error}")
                data = await resp.json()
                # #region agent log
                open(_log_path, "a").write(json.dumps({"hypothesisId": "A", "location": "fusion_agent_loop.py:_acquire_session:success", "message": "session_acquired", "data": {"session_id": data["session_id"], "total_duration_ms": int((_time.time() - _acq_start)*1000)}, "timestamp": int(_time.time()*1000)}) + "\n")
                # #endregion
                return data["session_id"]
    
    async def _execute_command(
        self,
        command: str,
        timeout: float = SWEREX_COMMAND_TIMEOUT,
        fetch_files: Optional[list[str]] = None,
    ) -> CommandResult:
        """Execute a command via the server.
        
        Args:
            command: The bash command to execute
            timeout: Command timeout in seconds
            fetch_files: Optional list of file paths to fetch after execution
            
        Returns:
            CommandResult with status, stdout, stderr, return_code, and optionally files
        """
        assert self.session_id is not None, "Session not acquired"
        
        client = await get_client()
        
        payload = {"command": command, "timeout": timeout}
        if fetch_files:
            payload["fetch_files"] = fetch_files
        
        async with client.post(
            f"{self.server_url}/session/{self.session_id}/execute",
            json=payload,
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                return CommandResult(
                    status="Failed",
                    stdout="",
                    stderr=f"HTTP error {resp.status}: {error}",
                    return_code=-1,
                    files=None,
                )
            data = await resp.json()
            return CommandResult(
                status=data["status"],
                stdout=data["stdout"],
                stderr=data["stderr"],
                return_code=data["return_code"],
                files=data.get("files"),
            )
    
    async def _release_session(self) -> None:
        """Release the session back to the server."""
        if not self.session_id:
            return
        
        try:
            client = await get_client()
            async with client.post(
                f"{self.server_url}/session/{self.session_id}/release"
            ) as resp:
                pass  # Ignore response on release
        except Exception as e:
            logger.warning(f"Failed to release session {self.session_id}: {e}")
        finally:
            self.session_id = None
    
    # =========================================================================
    # Agent Loop Methods
    # =========================================================================
    
    def flatten_structure(self, fs_list: list, prefix: str = "") -> dict:
        """Flatten nested file structure to flat dict of path -> base64 content."""
        files = {}
        for item in fs_list:
            path = f"{prefix}/{item['name']}" if prefix else item['name']
            if item['type'] == 'file':
                files[path] = base64.b64encode(item['content'].encode()).decode()
            else:
                files.update(self.flatten_structure(item['content'], path))
        return files
    
    def extract_bash_command(
        self, 
        text: str, 
        prefix: str = "<bash>", 
        suffix: str = "</bash>"
    ) -> Optional[str]:
        """Extract bash command from model output."""
        assert isinstance(text, str), f"text must be a string, got {type(text)}"
        
        # Skip thinking tags
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
    
    def _create_output_string(self, result: CommandResult) -> str:
        """Create output string from command result."""
        if result.status == "Success":
            return result.stdout
        else:
            if result.stderr:
                return f"Execution Failed: {result.stderr}"
            else:
                return f"Execution Failed: exit code {result.return_code}"
    
    def decode_fetched_files(self, files: Optional[dict[str, str]]) -> np.ndarray:
        """Decode base64-encoded files from server response.
        
        Args:
            files: Dict of filename -> base64-encoded content
            
        Returns:
            numpy array containing dict of filename -> decoded content
        """
        try:
            if not files:
                return np.array(dict())
            out_dict = {}
            for k, v in files.items():
                out_dict[k] = base64.b64decode(v).decode('utf-8')
            # Transform into numpy as DataProto expects arrays
            return np.array(out_dict)
        except Exception as e:
            logger.warning(f"Failed to decode files: {e}")
            return np.array(dict())
    
    async def execute_agent_command(self, command: str) -> tuple:
        """
        Execute a bash command with true statefulness.
        
        Unlike the deprecated FusionAgentLoop which replays all previous commands,
        this just executes the single command via the session server.
        State (cd, export, variables) naturally persists.
        
        Args:
            command: The bash command to execute
            
        Returns:
            Tuple of (output_string, fetched_files_array)
        """
        result = await self._execute_command(
            command,
            fetch_files=self.files_to_fetch if self.files_to_fetch else None,
        )
        output = self._create_output_string(result)
        fetched_files = self.decode_fetched_files(result.files)
        
        return output, fetched_files
    
    def _is_dangerous_command(self, cmd: str) -> bool:
        """Check if command contains dangerous patterns."""
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
        return any(pattern in cmd for pattern in dangerous_patterns)
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run the agent loop for one episode.
        
        This method:
        1. Acquires a session from the server
        2. Writes initial files and runs startup commands (via server)
        3. Runs the multi-turn agent loop
        4. Releases the session back to the server
        """
        # Parse tools_kwargs
        assert "tools_kwargs" in kwargs, "tools_kwargs required"
        tools_kwargs = json.loads(kwargs["tools_kwargs"])
        assert "files_dict" in tools_kwargs, "files_dict required in tools_kwargs"
        
        self.files_to_fetch = tools_kwargs.get("files_to_fetch", [])
        startup_commands = tools_kwargs.get("startup_commands", [])
        files_dict = tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"files_dict must be list, got {type(files_dict)}"
        self.files = self.flatten_structure(files_dict)
        
        # Acquire session from server
        self.session_id = await self._acquire_session(
            files=self.files if self.files else None,
            startup_commands=startup_commands if startup_commands else None,
        )
        
        try:
            return await self._run_episode(sampling_params, **kwargs)
        finally:
            # Always release session back to server
            await self._release_session()
    
    async def _run_episode(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the multi-turn agent loop for one episode."""
        # #region agent log
        import time as _time; import os as _os; _log_path = "./tmp/fusion_debug.log"; _os.makedirs(_os.path.dirname(_log_path), exist_ok=True); _episode_start = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "G_episode", "location": "fusion_agent_loop.py:_run_episode:start", "message": "episode_started", "data": {"session_id": self.session_id}, "timestamp": int(_time.time()*1000)}) + "\n")
        # #endregion
        messages = list(kwargs["raw_prompt"])
        conversation_messages = [dict(msg) for msg in messages]
        metrics = {}
        
        request_id = uuid4().hex
        num_turns = 0
        max_num_turns = self.config.actor_rollout_ref.rollout.multi_turn.get("max_assistant_turns", 5)
        mask = []
        
        # Tokenize initial prompt
        # #region agent log
        _tok_start = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "G_episode", "location": "fusion_agent_loop.py:_run_episode:tokenize_start", "message": "tokenize_starting", "data": {"session_id": self.session_id}, "timestamp": int(_time.time()*1000)}) + "\n")
        # #endregion
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        # #region agent log
        open(_log_path, "a").write(json.dumps({"hypothesisId": "G_episode", "location": "fusion_agent_loop.py:_run_episode:tokenize_done", "message": "tokenize_complete", "data": {"session_id": self.session_id, "tokenize_ms": int((_time.time() - _tok_start)*1000), "prompt_len": len(prompt_ids)}, "timestamp": int(_time.time()*1000)}) + "\n")
        # #endregion
        
        curr_input = list(prompt_ids)
        all_output_with_tool = []
        fetched_files = np.array({})
        
        with simple_timer("generate_sequences_all_turns", metrics):
            while num_turns < max_num_turns:
                assert len(curr_input) > 0
                assert isinstance(curr_input, list)
                assert isinstance(curr_input[-1], int)
                
                # Generate from LLM
                # #region agent log
                _llm_start = _time.time(); open(_log_path, "a").write(json.dumps({"hypothesisId": "G_episode", "location": "fusion_agent_loop.py:_run_episode:llm_start", "message": "llm_generate_starting", "data": {"session_id": self.session_id, "turn": num_turns, "input_len": len(curr_input)}, "timestamp": int(_time.time()*1000)}) + "\n")
                # #endregion
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=curr_input, sampling_params=sampling_params
                    )
                # #region agent log
                open(_log_path, "a").write(json.dumps({"hypothesisId": "G_episode", "location": "fusion_agent_loop.py:_run_episode:llm_done", "message": "llm_generate_complete", "data": {"session_id": self.session_id, "turn": num_turns, "llm_ms": int((_time.time() - _llm_start)*1000), "output_len": len(output.token_ids) if hasattr(output, 'token_ids') else 0}, "timestamp": int(_time.time()*1000)}) + "\n")
                # #endregion
                assert isinstance(output, TokenOutput), f"Expected TokenOutput, got {type(output)}"
                
                assert isinstance(output.token_ids, list)
                assert len(output.token_ids) > 0, "Empty output from LLM"
                
                all_output_with_tool += output.token_ids
                mask += [1] * len(output.token_ids)
                
                # Decode output
                decoded_output = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(output.token_ids)
                )
                
                conversation_messages.append({
                    "role": "assistant",
                    "content": decoded_output
                })
                
                # Extract bash command
                cmd = await self.loop.run_in_executor(
                    None,
                    lambda: self.extract_bash_command(decoded_output)
                )
                
                # If no command, exit loop
                if cmd is None:
                    break
                
                curr_input += output.token_ids
                
                # Execute command (with safety check)
                if self._is_dangerous_command(cmd):
                    cmd_output = f"bash: {cmd.split()[0]}: Operation not permitted"
                    fetched_files = np.array({})
                else:
                    with simple_timer("tool_calls", metrics):
                        cmd_output, fetched_files = await self.execute_agent_command(cmd)
                
                # Format tool response
                cmd_message = [{"role": "tool", "content": cmd_output}]
                conversation_messages.append({"role": "tool", "content": cmd_output})
                
                cmd_message_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        cmd_message, add_generation_prompt=True, tokenize=True, 
                        **self.apply_chat_template_kwargs
                    ),
                )
                
                curr_input += cmd_message_ids
                all_output_with_tool += cmd_message_ids
                mask += [0] * len(cmd_message_ids)
                
                if len(mask) >= self.response_length:
                    break
                
                num_turns += 1
        
        # Truncate to response_length
        assert len(mask) == len(all_output_with_tool)
        mask = mask[:self.response_length]
        all_output_with_tool = all_output_with_tool[:self.response_length]
        
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
