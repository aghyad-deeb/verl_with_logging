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
import requests
import numpy as np
from typing import Any, Optional
from uuid import uuid4

from recipe.fully_async_policy.agent_loop.agent_loop import AgentLoopOutput, FullyAsyncAgentLoopOutput
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

    def send_bash_command(self, code, files=dict(), files_to_fetch=[]):
        # print(f"{code=}")
        response = requests.post(self.url, json={
            'code': f'''{code}''',
            'language': 'bash',
            'run_timout': 1,
            'files': files,
            'fetch_files': files_to_fetch,
        })

        return response.json()

    def decode_fetched_files(self, resp_json):
        import base64
        try:
            out_dict = dict()
            if  "files" not in resp_json:
                return dict()
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

            # print(f"==================== state_script ====================\n\n")
            # print(state_script)
            # print(f"\n\n==================== end of state_script ====================\n\n")
            
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

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        check_server_running()

        print(f"\n\nin partial_fusion_agent_loop.run()\n\n")
        assert "tools_kwargs" in kwargs
        import json
        self.tools_kwargs=json.loads(kwargs["tools_kwargs"])
        assert  "files_dict" in self.tools_kwargs, f"{self.tools_kwargs=}"
        self.files_to_fetch = self.tools_kwargs.get("files_to_fetch", [])
        startup_commands = self.tools_kwargs.get("startup_commands", [])
        files_dict = self.tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"{files_dict=}"
        self.files = self.flatten_structure(files_dict)

        maybe_partial_output: Optional[FullyAsyncAgentLoopOutput] = kwargs.get("output", None)
        param_version = kwargs.get("param_version", 0)
        param_version_start = param_version
        param_version_end = param_version
        request_id = uuid4().hex
        fetched_files = np.array(dict())
        max_num_turns = self.config.actor_rollout_ref.rollout.multi_turn.get("max_assistant_turns", 5)

        if not maybe_partial_output:
            print(f"\n\nin 1 if\n\n")
            metrics = {}
            # empty command history for each run
            self.command_history = startup_commands
            num_turns = 0
            mask = list()
            messages = list(kwargs["raw_prompt"])
            print(f"\n\nbefore prompt_ids\n\n")
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            # print(f"{self.response_length=}")
            # Commented as response mask shouldn't include the prompt
            # mask += [0] * len(prompt_ids)
            all_output_with_tool = list()
            curr_input = [tok for tok in prompt_ids]
        else:
            if maybe_partial_output.is_cancel:
                print(f"\n\nin 2 if\n\n")
                metrics = maybe_partial_output.metrics
                param_version_start = maybe_partial_output.param_version_start
                self.command_history = maybe_partial_output.extra_fields["command_history"]
                num_turns = maybe_partial_output.num_turns
                mask = maybe_partial_output.response_mask
                prompt_ids = maybe_partial_output.prompt_ids
                all_output_with_tool = maybe_partial_output.response_ids
                curr_input = (
                    maybe_partial_output.prompt_ids
                    + maybe_partial_output.response_ids
                )
            else:
                print(f"\n\nin 3 if\n\n")
                return maybe_partial_output

        print(f"\n\nbefore start of loop\n\n")
        with simple_timer("generate_sequences_all_turns", metrics):
            while num_turns < max_num_turns:
                # Use processor if available for multimodal support
                assert len(curr_input) > 0
                assert isinstance(curr_input, list)
                assert isinstance(curr_input[-1], int)
                # print(f"{num_turns=}, {self.tokenizer.decode(curr_input)=}")

                print(f"\n\nbefore generate_for_partial\n\n")
                # if len(curr_input) > 
                token_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                    request_id=request_id, prompt_ids=curr_input, sampling_params=sampling_params
                )
                #! This will fail but we'll get to know the type
                # assert isinstance(token_ids, TokenOutput), f"{type(output)=}, {output=}"

                assert isinstance(token_ids, list)
                assert isinstance(token_ids[0], int) #! will fail if len is 0, but shouldn't ever be
                all_output_with_tool += token_ids
                mask += [1] * len(token_ids)
                print(f"\n\nbefore decode\n\n")

                decoded_output = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.decode(token_ids)
                )
                cmd = await self.loop.run_in_executor(
                        None,
                        lambda: self.extract_bash_command(decoded_output)
                )
                # print(f"{cmd=}, {decoded_output=}")
                # if agent doesn't output a command, we exit the loop
                if cmd is None:
                    # print(f"\nbreaking as cmd is None\n")
                    break

                curr_input += token_ids

                print(f"\n\nbefore execute_command\n\n")
                cmd_output, fetched_files = await self.loop.run_in_executor(
                        None,
                        lambda: self.execute_agent_command(cmd)
                )
                cmd_message = [{
                    "role": "tool",
                    "content": cmd_output
                }]
                print(f"\n\nbefore run_in_executor\n\n")
                cmd_message_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        cmd_message, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
                curr_input += cmd_message_ids
                all_output_with_tool += cmd_message_ids
                mask += [0] * len(cmd_message_ids)
                if len(mask) >= self.response_length or is_cancel:
                    break

                num_turns += 1
                
        print(f"\n\nfinished while loop, {is_cancel=}\n\n")
        # response_mask = [1] * len(output.token_ids)
        assert len(mask) == len(all_output_with_tool), f"{len(mask)=}, {len(all_output_with_tool)=}, {mask=}\n{all_output_with_tool=}"
        mask = mask[: self.response_length]
        all_output_with_tool = all_output_with_tool[: self.response_length]
        assert len(mask) == len(all_output_with_tool), f"{len(mask)=}, {len(all_output_with_tool)=}, {mask=}\n{all_output_with_tool=}"

        output = FullyAsyncAgentLoopOutput(
            prompt_ids=prompt_ids[:self.prompt_length],
            # prompt_ids=,
            response_ids=all_output_with_tool[: self.response_length],
            # response_ids=, #! I don't think I want these here
            response_mask=mask[: self.response_length],
            # response_mask=response_mask[: self.response_length],
            response_logprobs=log_probs[: self.response_length] if output.log_probs else None,
            # response_logprobs=,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=dict(
                fetched_files=fetched_files,
                command_history=self.command_history,
            ),
            is_cancel=is_cancel,
            param_version_start=param_version_start,
            param_version_end=param_version_end,
        )
        return output
