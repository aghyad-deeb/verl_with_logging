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
import copy
import logging
import os
import base64
import requests
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
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




@register("fusion_agent_loop")
class FusionAgentLoop(AgentLoopBase):
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
            'files': files,
            'fetch_files': files_to_fetch,
        })

        return response.json()

    def decode_fetched_files(self, resp_json):
        import base64
        out_dict = dict()
        if  "files" not in resp_json:
            return dict()
        for k, v in resp_json["files"].items():
            out_dict[k] = base64.b64decode(v).decode('utf-8')
        # transform into numpy as DataProto expects arrays
        import numpy as np
        return np.array(out_dict)

    def create_command_output(self, result):
        if result["status"] == "Success":
            return f"<output>{result['run_result']['stdout']}</output>"
        else:
            if "run_result" in result and "stderr" in result["run_result"]:
                return f"<output>Execution Failed: {result['run_result']['stderr']}</output>"
            else:
                print(f"\n\n\n\nExecution failed without std Err: {result=}\n\n\n\n")
                return f"<output>Execution Failed: {result=}"

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

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        check_server_running()

        assert "tools_kwargs" in kwargs
        import json
        self.tools_kwargs=json.loads(kwargs["tools_kwargs"])
        assert  "files_dict" in self.tools_kwargs, f"{self.tools_kwargs=}"
        self.files_to_fetch = self.tools_kwargs.get("files_to_fetch", {})
        files_dict = self.tools_kwargs["files_dict"]
        assert isinstance(files_dict, list), f"{files_dict=}"
        self.files = self.flatten_structure(files_dict)

        messages = list(kwargs["raw_prompt"])
        metrics = {}
        # empty command history for each run
        self.command_history = list()
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
        # print(f"{self.response_length=}")
        # Commented as response mask shouldn't include the prompt
        # mask += [0] * len(prompt_ids)
        curr_input = [tok for tok in prompt_ids]
        all_output_with_tool = list()
        import numpy as np
        fetched_files = np.arary(dict())
        with simple_timer("generate_sequences_all_turns", metrics):
            while num_turns < max_num_turns:
                # Use processor if available for multimodal support
                assert len(curr_input) > 0
                assert isinstance(curr_input, list)
                assert isinstance(curr_input[-1], int)
                # print(f"{num_turns=}, {self.tokenizer.decode(curr_input)=}")

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
                decoded_output = self.tokenizer.decode(output.token_ids)
                cmd = self.extract_bash_command(decoded_output)
                # print(f"{cmd=}, {decoded_output=}")
                # if agent doesn't output a command, we exit the loop
                if cmd is None:
                    # print(f"\nbreaking as cmd is None\n")
                    break

                curr_input += output.token_ids

                cmd_output, fetched_files = self.execute_agent_command(cmd)
                cmd_message = [{
                    "role": "tool",
                    "content": cmd_output
                }]
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
        import random
        if random.random() < 0.01:
            print(f"{self.tokenizer.decode(all_output_with_tool)=}")

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
            )
        )
        return output