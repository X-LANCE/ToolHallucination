import re
import os
import json
import time
import requests
from tqdm import tqdm
from termcolor import colored
import random
from toolbench.inference.LLM.chatgpt_function_model import ChatGPTFunction, chat_completion_request
from toolbench.inference.LLM.davinci_model import Davinci
from toolbench.inference.LLM.tool_llama_lora_model import ToolLLaMALoRA
from toolbench.inference.LLM.tool_llama_model import ToolLLaMA
from toolbench.inference.LLM.tool_llama_vllm_model import ToolLLaMA_vllm
from toolbench.inference.LLM.llama3_sft_model import Llama3Model
from toolbench.inference.LLM.qwen2_sft_model import Qwen2Model
from toolbench.inference.LLM.retriever import ToolRetriever
from toolbench.inference.Algorithms.single_chain import single_chain
from toolbench.inference.Algorithms.DFS import DFS_tree_search
from toolbench.inference.server import get_rapidapi_response
from toolbench.utils import (
    standardize,
    change_name,
    replace_llama_with_condense
)
import traceback

from toolbench.inference.Downstream_tasks.base_env import base_env
from concurrent.futures import ThreadPoolExecutor

import pdb


# For pipeline environment preparation
def get_white_list(tool_root_dir):
    # print(tool_root_dir)
    white_list_dir = os.path.join(tool_root_dir)
    white_list = {}
    for cate in tqdm(os.listdir(white_list_dir)):
        if not os.path.isdir(os.path.join(white_list_dir,cate)):
            continue
        for file in os.listdir(os.path.join(white_list_dir,cate)):
            if not file.endswith(".json"):
                continue
            standard_tool_name = file.split(".")[0]
            # print(standard_tool_name)
            with open(os.path.join(white_list_dir,cate,file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data["tool_name"]
            white_list[standardize(origin_tool_name)] = {"description": js_data["tool_description"], "standard_tool_name": standard_tool_name}
    return white_list

def contain(candidate_list, white_list):
    output = []
    for cand in candidate_list:
        if cand not in white_list.keys():
            return False
        output.append(white_list[cand])
    return output


# rapidapi env wrapper
class rapidapi_wrapper(base_env):
    def __init__(self, query_json, tool_descriptions, retriever, args, process_id=0):
        super(rapidapi_wrapper).__init__()

        self.tool_root_dir =  os.getenv("TOOL_ROOT_DIR", None)
        self.toolbench_key = os.getenv("TOOLBENCH_KEY", None)
        self.rapidapi_key = args.rapidapi_key
        self.use_rapidapi_key = args.use_rapidapi_key
        self.api_customization = args.api_customization
        self.service_url = os.getenv("SERVICE_URL", "http://8.130.32.149:8080/rapidapi")
        self.max_observation_length = args.max_observation_length
        self.observ_compress_method = args.observ_compress_method
        self.retriever = retriever
        self.process_id = process_id
        self.openai_key =  os.getenv("OPENAI_KEY", None)
        self.openai_base_url =  os.getenv("OPENAI_API_BASE", None)

        self.tool_names = []
        self.cate_names = []

        self.input_description = query_json["query"]
        self.functions = []
        self.api_name_reflect = {}
        orig_user_query = query_json["orig_query"] if 'orig_query' in query_json else query_json["query"]
        self.user_simulator_history = \
            [{"role": "system", "content": f"You are a user simulator. You present a problem-solving query to the assistant,"
                                           f" and you need to play the role of the user to answer the assistant's questions."
                                           f" The information you know is only in the user query. When the assistant asks a"
                                           f" question that can be answered based on the information in the user query, "
                                           f"you should provide the answer. If the user query does not contain the relevant"
                                           f" information, you should tell the assistant that you don't know.\nUser Query:\n{orig_user_query}"}]
        # pdb.set_trace()

        if self.retriever is not None:
            query_json = self.retrieve_rapidapi_tools(self.input_description, args.retrieved_api_nums, self.tool_root_dir)
            data_dict = self.fetch_api_json(query_json)
            tool_descriptions = self.build_tool_description(data_dict)
        else:
            data_dict = self.fetch_api_json(query_json)
            if len(data_dict["api_list"]) != len(tool_descriptions):
                tool_descriptions = self.build_tool_description(data_dict)

        for k,api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_descriptions[k][0]
            openai_function_json,cate_name, pure_api_name = self.api_json_to_openai_json(api_json,standard_tool_name)
            self.functions.append(openai_function_json)
            self.api_name_reflect[openai_function_json["function"]["name"]] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)

        finish_func = {
            "type": "function",
            "function": {
                "name": "Finish",
                "description": "If you believe that you have obtained a result that can answer the task, "
                               "please call this function to provide the final answer. "
                               "Alternatively, if you recognize that you are unable to proceed with the task "
                               "in the current state or current tools cannot satisfy the user's query or you need to ask the user questions to obtain certain essential information required to complete the task, "
                               "call this function to restart or change tools or talk to user. "
                               "Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "return_type": {
                            "type": "string",
                            "enum": ["give_answer","give_up_and_restart", "give_up_and_change_tools", "give_up_and_talk_to_user"],
                        },
                        "final_answer": {
                            "type": "string",
                            "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"",
                        }
                    },
                    "required": ["return_type"],
                    "optional": ["final_answer"],
                },
            }
        }

        self.functions.append(finish_func)
        self.CALL_MAX_TIME = 3
        self.task_description = f'''You should use functions to help handle the real time user querys. Remember:
1.The tools provided to you are limited. You need to carefully discern the purpose of each tool. If you believe that the purpose of tools cannot satisfy the user's query, please directly call the function Finish->give_up_and_change_tools.
2.Some tool invocation parameters might not be present in the user's query. You need to determine if the essential information (the tool invocation parameters) required to complete the task are already included in the query. If they are not, please directly call the function Finish->give_up_and_talk_to_user.
3.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
4.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:\n'''
        
        unduplicated_reflection = {}
        for standardize_tool_name, tool_des in tool_descriptions:
            unduplicated_reflection[standardize_tool_name] = tool_des

        for k,(standardize_tool_name, tool_des) in enumerate(unduplicated_reflection.items()):
            try:
                striped = tool_des[:512].replace('\n','').strip()
            except:
                striped = ""
            if striped == "":
                striped = "None"
            self.task_description += f"{k+1}.{standardize_tool_name}: {striped}\n"

        self.success = 0

    def build_tool_description(self, data_dict):
        white_list = get_white_list(self.tool_root_dir)
        origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
        tool_des = contain(origin_tool_names,white_list)
        tool_descriptions = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
        return tool_descriptions
    
    def retrieve_rapidapi_tools(self, query, top_k, jsons_path):
        retrieved_tools = self.retriever.retrieving(query, top_k=top_k)
        query_json = {"api_list":[]}
        for tool_dict in retrieved_tools:
            if len(query_json["api_list"]) == top_k:
                break
            category = tool_dict["category"]
            tool_name = tool_dict["tool_name"]
            api_name = tool_dict["api_name"]
            if os.path.exists(jsons_path):
                if os.path.exists(os.path.join(jsons_path, category)):
                    if os.path.exists(os.path.join(jsons_path, category, tool_name+".json")):
                        query_json["api_list"].append({
                            "category_name": category,
                            "tool_name": tool_name,
                            "api_name": api_name
                        })
        return query_json

    def run_user_simulator(self, question):
        self.user_simulator_history.append({'role': 'user', 'content': question})
        response = chat_completion_request(self.openai_key, self.openai_base_url, self.user_simulator_history,
                                model="gpt-4o-mini", temperature=0)
        try:
            total_tokens = response['usage']['total_tokens']
            message = response["choices"][0]["message"]['content']

            if self.process_id == 0:
                print(f"[process({self.process_id})]total tokens: {total_tokens}")
            self.user_simulator_history.append({'role': 'assistant', 'content': message})
            return {"error": "", "response": f"{message}"}
        except BaseException as e:
            print(f"[process({self.process_id})]Parsing Exception: {repr(e)}. Try again.")
            traceback.print_exc()
            if response is not None:
                print(f"[process({self.process_id})]OpenAI return: {response}")
            pdb.set_trace()
            return {"error": f"{repr(e)}", "response": f"{response['error']}"}
    
    def fetch_api_json(self, query_json):
        data_dict = {"api_list":[]}
        for item in query_json["api_list"]:
            cate_name = item["category_name"]
            tool_name = standardize(item["tool_name"])
            api_name = change_name(standardize(item["api_name"]))
            tool_json = json.load(open(os.path.join(self.tool_root_dir, cate_name, tool_name + ".json"), "r"))
            append_flag = False
            api_dict_names = []
            for api_dict in tool_json["api_list"]:
                api_dict_names.append(api_dict["name"])
                pure_api_name = change_name(standardize(api_dict["name"]))
                if pure_api_name != api_name:
                    continue
                api_json = {}
                api_json["category_name"] = cate_name
                api_json["api_name"] = api_dict["name"]
                api_json["api_description"] = api_dict["description"]
                api_json["required_parameters"] = api_dict["required_parameters"]
                api_json["optional_parameters"] = api_dict["optional_parameters"]
                api_json["tool_name"] = tool_json["tool_name"]
                data_dict["api_list"].append(api_json)
                append_flag = True
                break
            if not append_flag:
                print(api_name, api_dict_names)
        return data_dict


    def api_json_to_openai_json(self, api_json,standard_tool_name):
        description_max_length=256
        function_templete = {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [""],
                    "optional": [""],
                }
            }
        }
        templete = function_templete['function']
        
        map_type = {
            "NUMBER": "integer",
            "STRING": "string",
            "BOOLEAN": "boolean"
        }

        pure_api_name = change_name(standardize(api_json["api_name"]))
        templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
        templete["name"] = templete["name"][-64:]

        templete["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."
        
        if api_json["api_description"].strip() != "":
            tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],templete['name'])[:description_max_length]
            templete["description"] = templete["description"] + f"The description of this function is: \"{tuncated_description}\""
        if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
            for para in api_json["required_parameters"]:
                name = standardize(para["name"])
                name = change_name(name)
                if para["type"] in map_type:
                    param_type = map_type[para["type"]]
                else:
                    param_type = "string"
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                }

                default_value = para['default']
                if len(str(default_value)) != 0:    
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length],
                        "example_value": default_value
                    }
                else:
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length]
                    }

                templete["parameters"]["properties"][name] = prompt
                templete["parameters"]["required"].append(name)
            for para in api_json["optional_parameters"]:
                name = standardize(para["name"])
                name = change_name(name)
                if para["type"] in map_type:
                    param_type = map_type[para["type"]]
                else:
                    param_type = "string"

                default_value = para['default']
                if len(str(default_value)) != 0:    
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length],
                        "example_value": default_value
                    }
                else:
                    prompt = {
                        "type":param_type,
                        "description":para["description"][:description_max_length]
                    }

                templete["parameters"]["properties"][name] = prompt
                templete["parameters"]["optional"].append(name)

        return function_templete, api_json["category_name"],  pure_api_name

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def step(self,**args):
        obs, code = self._step(**args)
        if len(obs) > self.max_observation_length:
            obs = obs[:self.max_observation_length] + "..."
        return obs, code

    def _step(self, action_name="", action_input=""):
        """Need to return an observation string and status code:
            0 means normal response
            1 means there is no corresponding api name
            2 means there is an error in the input
            3 represents the end of the generation and the final answer appears
            4 means that the model decides to pruning by itself
            5 represents api call timeout
            6 for 404
            7 means not subscribed
            8 represents unauthorized
            9 represents too many requests
            10 stands for rate limit
            11 message contains "error" field
            12 error sending request
        """
        if action_name == "Finish":
            try:
                json_data = json.loads(action_input,strict=False)
            except:
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = "give_answer"
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = "give_up_and_restart"
                    elif '"return_type": "give_up_and_change_tools"' in action_input:
                        return_type = "give_up_and_change_tools"
                    elif '"return_type": "give_up_and_talk_to_user"' in action_input:
                        return_type = "give_up_and_talk_to_user"
                    else:
                        return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
                    json_data["return_type"] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
                    json_data["final_answer"] = final_answer
            if "return_type" not in json_data.keys():
                return "{error:\"must have \"return_type\"\"}", 2
            if json_data["return_type"] == "give_up_and_restart":
                return "{\"response\":\"chose to give up and restart\"}",4
            elif json_data["return_type"] == "give_up_and_change_tools":
                return "{\"response\":\"chose to give up and change tools\"}", 4
            elif json_data["return_type"] == "give_up_and_talk_to_user":
                return "{\"response\":\"chose to give up and talk to user\"}", 4
            elif json_data["return_type"] == "give_answer":
                if "final_answer" not in json_data.keys():
                    return "{error:\"must have \"final_answer\"\"}", 2
                
                self.success = 1 # succesfully return final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                return "{error:\"\"return_type\" is not a valid choice\"}", 2
        elif action_name == "TalkToUser":
            user_resposne = self.run_user_simulator(action_input)
            return json.dumps(user_resposne), 0
        elif action_name == "":
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1
        else:

            for k, function_dict in enumerate(self.functions):
                function = function_dict['function']
                # import pdb; pdb.set_trace()
                if function["name"].endswith(action_name):
                    pure_api_name = self.api_name_reflect[function["name"]]
                    payload = {
                        "category": self.cate_names[k],
                        "tool_name": self.tool_names[k],
                        "api_name": pure_api_name,
                        "tool_input": action_input,
                        "strip": self.observ_compress_method,
                        "toolbench_key": self.toolbench_key
                    }
                    if self.process_id == 0:
                        print(colored(f"query to {self.cate_names[k]}-->{self.tool_names[k]}-->{action_name}",color="yellow"))
                    if self.use_rapidapi_key or self.api_customization:
                        payload["rapidapi_key"] = self.rapidapi_key
                        response = get_rapidapi_response(payload, api_customization=self.api_customization)
                    else:
                        time.sleep(2) # rate limit: 30 per minute
                        headers = {"toolbench_key": self.toolbench_key}
                        timeout = None if self.service_url.endswith("virtual") else 15
                        try:
                            response = requests.post(self.service_url, json=payload, headers=headers, timeout=timeout)
                        except requests.exceptions.Timeout:
                            return json.dumps({"error": f"Timeout error...", "response": ""}), 5
                        if response.status_code != 200:
                            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
                        try:
                            response = response.json()
                        except:
                            print(response)
                            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
                    # 1 Hallucinating function names
                    # 4 means that the model decides to pruning by itself
                    # 5 represents api call timeout
                    # 6 for 404
                    # 7 means not subscribed
                    # 8 represents unauthorized
                    # 9 represents too many requests
                    # 10 stands for rate limit
                    # 11 message contains "error" field
                    # 12 error sending request
                    if response["error"] == "API not working error...":
                        status_code = 6
                    elif response["error"] == "Unauthorized error...":
                        status_code = 7
                    elif response["error"] == "Unsubscribed error...":
                        status_code = 8
                    elif response["error"] == "Too many requests error...":
                        status_code = 9
                    elif response["error"] == "Rate limit per minute error...":
                        print("Reach api calling limit per minute, sleeping...")
                        time.sleep(10)
                        status_code = 10
                    elif response["error"] == "Message error...":
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code
                    # except Exception as e:
                    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1


class pipeline_runner:
    def __init__(self, args, add_retrieval=False, process_id=0, server=False):
        self.args = args
        self.add_retrieval = add_retrieval
        self.process_id = process_id
        self.server = server
        self.tool_root_dir = os.getenv("TOOL_ROOT_DIR", None)
        # if not self.server: self.task_list = self.generate_task_list()
        # else: self.task_list = []
        self.task_list = []

    def get_backbone_model(self):
        args = self.args
        if args.backbone_model == "toolllama":
            # ratio = 4 means the sequence length is expanded by 4, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
            ratio = int(args.max_sequence_length/args.max_source_sequence_length)
            if ratio > 1:
                replace_llama_with_condense(ratio=ratio)
            if args.lora:
                backbone_model = ToolLLaMALoRA(base_name_or_path=args.model_path, model_name_or_path=args.lora_path, max_sequence_length=args.max_sequence_length)
            else:
                backbone_model = ToolLLaMA(model_name_or_path=args.model_path, max_sequence_length=args.max_sequence_length)
        # elif args.backbone_model == "llama3.1":
        #     backbone_model = ToolLLaMA(model_name_or_path=args.model_path,
        #                                max_sequence_length=args.max_sequence_length,
        #                                template="llama3.1-single-round",
        #                                do_sampling=args.do_sampling)
        else:
            backbone_model = args.backbone_model
        return backbone_model

    def get_retriever(self):
        return ToolRetriever(corpus_tsv_path=self.args.corpus_tsv_path, model_path=self.args.retrieval_model_path)

    def get_args(self):
        return self.args

    def generate_task_list(self):
        args = self.args
        query_dir = args.input_query_file
        answer_dir = args.output_answer_file
        if not os.path.exists(answer_dir):
            os.makedirs(answer_dir, exist_ok=True)
        method = args.method
        backbone_model = self.get_backbone_model()
        white_list = get_white_list(self.tool_root_dir)
        task_list = []
        querys = json.load(open(query_dir, "r"))
        for query_id, data_dict in enumerate(querys):
            if "query_id" in data_dict:
                query_id = data_dict["query_id"]
            if "api_list" in data_dict:
                origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
                tool_des = contain(origin_tool_names,white_list)
                if tool_des == False:
                    continue
                tool_des = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
            else:
                tool_des = None
            task_list.append((method, backbone_model, query_id, data_dict, args, answer_dir, tool_des))
        return task_list

    def generate_qid_list(self):
        args = self.args
        query_dir = args.input_query_file
        answer_dir = args.output_answer_file
        if not os.path.exists(answer_dir):
            os.makedirs(answer_dir, exist_ok=True)
        method = args.method
        task_list = []
        querys = json.load(open(query_dir, "r"))
        for query_id, data_dict in enumerate(querys):
            if "query_id" in data_dict:
                query_id = data_dict["query_id"]
            task_list.append((method, query_id, data_dict, args, answer_dir))
        return task_list
    
    def method_converter(self, backbone_model, openai_key, method, env, process_id, single_chain_max_step=12, max_query_count=60, callbacks=None):
        if callbacks is None: callbacks = []
        args = self.args
        do_sampling = args.do_sampling
        #print(f"do sampling: {self.args.do_sampling}")
        if backbone_model == "chatgpt_function":
            # model = "gpt-3.5-turbo-16k-0613"
            # model = os.getenv('CHAT_MODEL', "gpt-3.5-turbo-16k-0613")
            base_url = os.getenv('OPENAI_API_BASE', None)
            llm_forward = ChatGPTFunction(model=self.args.chatgpt_model, openai_key=openai_key, base_url=base_url)
        elif backbone_model == "llama3":
            base_url = os.getenv('VLLM_API_BASE', None)
            llm_forward = Llama3Model(model=self.args.model_path, openai_key=openai_key, base_url=base_url, do_sampling=do_sampling)
        elif backbone_model == "qwen2":
            base_url = os.getenv('VLLM_API_BASE', None)
            llm_forward = Qwen2Model(model=self.args.model_path, openai_key=openai_key, base_url=base_url, do_sampling=do_sampling)
        elif backbone_model == "ToolLLaMA_vllm":
            base_url = os.getenv('VLLM_API_BASE', None)
            print("base_url", base_url)
            llm_forward = ToolLLaMA_vllm(model=self.args.model_path, openai_key=openai_key, base_url=base_url, do_sampling=do_sampling)
        elif backbone_model == "davinci":
            model = os.getenv('CHAT_MODEL', "gpt-3.5-turbo-16k-0613")
            base_url = os.getenv('OPENAI_API_BASE', None)
            llm_forward = Davinci(model=model, openai_key=openai_key)
        else:
            model = backbone_model
            llm_forward = model
        
        if method.startswith("CoT"):
            passat = int(method.split("@")[-1])
            chain = single_chain(llm=llm_forward, io_func=env,process_id=process_id)
            result = chain.start(
                                pass_at=passat,
                                single_chain_max_step=single_chain_max_step,
                                answer=1)
        elif method.startswith("DFS"):
            pattern = r".+_w(\d+)"
            re_result = re.match(pattern,method)
            assert re_result != None
            width = int(re_result.group(1))
            with_filter = True
            if "woFilter" in method:
                with_filter = False
            chain = DFS_tree_search(llm=llm_forward, io_func=env,process_id=process_id, callbacks=callbacks)
            result = chain.start(
                                single_chain_max_step=single_chain_max_step,
                                tree_beam_size = width,
                                max_query_count = max_query_count,
                                answer=1,
                                with_filter=with_filter)
        else:
            print("invalid method")
            raise NotImplementedError
        return chain, result
    
    def run_single_task(self, method, backbone_model, query_id, data_dict, args, output_dir_path, tool_des, retriever=None, process_id=0, callbacks=None, server= None):
        if server is None:
            server = self.server
        if callbacks is None:
            if server: print("Warning: no callbacks are defined for server mode")
            callbacks = []
        splits = output_dir_path.split("/")
        os.makedirs("/".join(splits[:-1]),exist_ok=True)
        os.makedirs("/".join(splits),exist_ok=True)
        output_file_path = os.path.join(output_dir_path,f"{query_id}_{method}.json")
        if (not server) and os.path.exists(output_file_path):
            return
        [callback.on_tool_retrieval_start() for callback in callbacks]
        env = rapidapi_wrapper(data_dict, tool_des, retriever, args, process_id=process_id)
        [callback.on_tool_retrieval_end(
            tools=env.functions
        ) for callback in callbacks]
        query = data_dict["query"]
        if process_id == 0:
            print(colored(f"[process({process_id})]now playing {query}, with {len(env.functions)} APIs", "green"))
        [callback.on_request_start(
            user_input=query,
            method=method,
        ) for callback in callbacks]
        chain,result = self.method_converter(
            backbone_model=backbone_model,
            openai_key=os.getenv("OPENAI_KEY", None),
            method=method,
            env=env,
            process_id=process_id,
            single_chain_max_step=args.single_chain_max_step,
            # max_query_count=200,
            max_query_count=args.max_query_count,
            callbacks=callbacks
        )
        [callback.on_request_end(
            chain=chain.terminal_node[0].messages,
            outputs=chain.terminal_node[0].description,
        ) for callback in callbacks]
        if output_dir_path is not None:
            try:
                with open(output_file_path,"w") as writer:
                    data = chain.to_json(answer=True,process=True)
                    data["answer_generation"]["query"] = query
                    json.dump(data, writer, indent=2)
                    success = data["answer_generation"]["valid_data"] and "give_answer" in data["answer_generation"]["final_answer"]
                    if process_id == 0:
                        print(colored(f"[process({process_id})]valid={success}", "green"))
            except:
                print(f'File dumped failed.')
                print(data)
        return result
        
    def run(self):
        qid_list = self.generate_qid_list()
        new_qid_list = []
        for task in qid_list:
            out_dir_path = task[-1]
            query_id = task[1]
            output_file_path = os.path.join(out_dir_path,f"{query_id}_{self.args.method}.json")
            if not os.path.exists(output_file_path):
                new_qid_list.append(task[1])
        print(f"total tasks: {len(qid_list)} undo tasks: {len(new_qid_list)}")
        if len(new_qid_list) == 0:
            return 0

        self.task_list = self.generate_task_list()
        self.task_list = sorted(self.task_list, key=lambda x: x[2])
        task_list = self.task_list
        random.seed(self.args.random_seed)
        random.shuffle(task_list)
        new_task_list = []
        for task in task_list:
            out_dir_path = task[-2]
            query_id = task[2]
            output_file_path = os.path.join(out_dir_path,f"{query_id}_{self.args.method}.json")
            if not os.path.exists(output_file_path):
                new_task_list.append(task)
        task_list = new_task_list
        task_list = sorted(task_list, key=lambda x: x[2])

        def _split_list(lst, num_parts, part_index):
            if part_index < 0 or part_index >= num_parts:
                raise ValueError("part_index not valid")

            part_size = len(lst) // num_parts
            remainder = len(lst) % num_parts

            start = part_index * part_size + min(part_index, remainder)
            end = start + part_size + (1 if part_index < remainder else 0)

            return lst[start:end]

        print(f"undo tasks: {len(task_list)}")

        if len(task_list) == 0:
            return 0

        if self.args.task_split_num > 0:
            split_task_list = _split_list(self.task_list, self.args.task_split_num, self.args.task_split_index)
            filtered_task_list = []
            for task in split_task_list:
                query_id = task[2]
                if query_id in new_qid_list:
                    filtered_task_list.append(task)
            task_list = filtered_task_list
            if len(task_list) == 0:
                return 0
            print(f"doing {len(task_list)} tasks: task qid\n {[task[2] for task in task_list]}")



        if self.add_retrieval:
            retriever = self.get_retriever()
        else:
            retriever = None
        if self.args.num_thread == 1:
            for k, task in enumerate(task_list):
                print(f"process[{self.process_id}] doing task {k}/{len(task_list)}: real_task_id_{task[2]}")
                result = self.run_single_task(*task, retriever=retriever, process_id=self.process_id)
        else:
            def distribute_single_tasks(input):
                id, task = input
                return self.run_single_task(*task, retriever=retriever, process_id=id + self.process_id)

            with ThreadPoolExecutor(self.args.num_thread) as executor:
                for _ in tqdm(
                    executor.map(distribute_single_tasks, zip(range(len(task_list)), task_list)), 
                    total=len(task_list), 
                    disable=self.args.disable_tqdm
                ):
                    pass
