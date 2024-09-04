import copy
import re

from openai_schema import ChatMessage, ImageMessageContent
from typing import Dict, List, Literal, Optional, Union, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM    # 加载模型和标记器
from transformers.generation import GenerationConfig

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()

"""
Load model and tokenizer by Hugging Face framework.
"""
def load_model(_model_path: str, device_map: Literal["cuda", "cpu", "auto"] = "auto", trust_remote_code: bool = True
               ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    _model = AutoModelForCausalLM.from_pretrained(
        _model_path,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        resume_download=True)
    
    _tokenizer = AutoTokenizer.from_pretrained(
        _model_path,
        trust_remote_code=trust_remote_code,
        resume_download=True)
    
    _model.generation_config = GenerationConfig.from_pretrained(
        _model_path,
        trust_remote_code=trust_remote_code,
        resume_download=True)
    
    return _model, _tokenizer

# 旨在处理 stop_words 列表中的元素，以便在进行文本处理时能够避免不必要的前导换行符 \n 并且去重
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words

# 从给定的文本响应中去除指定的停止词
def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response

def _sort_list(_data: List[ImageMessageContent]):
    """按类型排序列表, 用以修复: https://github.com/QwenLM/Qwen-VL/issues/164"""
    return sorted(_data, key=lambda item: item.type not in ['image', 'box'])


"""
Convert and create the query for the model.chat function.
"""
def _create_query(_query: ChatMessage, **kwargs):
    if isinstance(_query.content, str):
        return [{"text": _query.content}]
    else:
        _query_list = []
        for index, content in enumerate(_sort_list(_query.content)):
            if content.type == "text":
                _query_list.append({"text": content.text})
            elif content.type == "image":
                # @TODO processing images
                # print(f"Save Image, path: {_img_path}")
                _query_list.append({"image": _img_path})
    return _query_list


def parse_messages(messages:List[ChatMessage], functions):
    # 检查是否有用户消息
    if all(m.role != "user" for m in messages):
        raise ValueError("Expecting at least one user message.")
    
    # 初始化和处理系统消息: 如果消息列表的第一个消息是系统消息，则提取并处理该消息。如果系统消息与默认消息相同，则将其清空。
    messages = copy.deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0].role == "system":
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""
            
    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")                # 函数的名称
            name_m = func_info.get("name_for_model", name)  # 函数在模型中的名称
            name_h = func_info.get("name_for_human", name)  # 函数在用户界面中的名称
            desc = func_info.get("description", "")         # 函数的描述
            desc_m = func_info.get("description_for_model", desc)   # 函数在模型中的描述
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )       # 格式化函数描述
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()
        
    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal answer: ",
        "zh": "\nThought: 我会作答了。\nFinal answer: ",
    }
    
    _messages = messages
    messages = []
    for msg_idx, msg in enumerate(_messages):
        role, content, func_call = msg.role, msg.content, msg.function_call
        
        if content:
            if isinstance(content, str):
                content = content.lstrip("\n").rstrip()
            else:
                _content = ""
                for _idx, _c in enumerate(_sort_list(content)):
                    if _c.type == "text":
                        _content += _c.text.lstrip("\n").rstrip()
                    elif _c.type == "image":
                        _content += ("<img>" + _c.image + "</img>").lstrip("\n").rstrip()
                content = _content
            print(f"content: {content}")
            
        # 处理 function 角色
        if role == "function":
            # 如果角色是 function，确保之前的消息是 assistant 角色，否则抛出错误
            if (len(messages) == 0) or (messages[-1].role != "assistant"):
                raise ValueError("Expecting role assistant before role function.")
            
            # 将 function 消息的内容追加到前一条 assistant 消息的 content 字段中，格式为 \nObservation: {content}
            messages[-1].content += f"\nObservation: {content}"
            # 如果当前消息是 _messages 列表中的最后一条消息，还会在 assistant 消息内容后添加 \nThought:
            if msg_idx == len(_messages) - 1:
                messages[-1].content += "\nThought:"
                
            # 处理 assistant 角色:
        elif role == "assistant":
            # 如果角色是 assistant，确保消息列表中有 user 角色的消息，否则抛出错误
            if len(messages) == 0:
                raise ValueError("Expecting role user before role assistant.")
            last_msg = messages[-1].content
            # 检查前一条消息的内容是否包含中文
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0
            # 如果没有函数调用, 且 functions 列表存在
            if func_call is None:
                if functions:
                    # 根据消息是否包含中文选择适当的 dummy_thought
                    content = dummy_thought["zh" if last_msg_has_zh else "en"] + content
            else:
                # 如果有函数调用，根据 func_call 的内容构建相应的 content
                f_name, f_args = func_call["name"], func_call["arguments"]
                # 如果 content 为空，添加默认的思考内容和行动描述
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."
                content = f"\n{content}\nAction: {f_name}\nAction Input: {f_args}"
            # 如果前一条消息的角色是 user，则将 assistant 消息作为新的条目添加
            if messages[-1].role == "user":
                messages.append(
                    ChatMessage(role="assistant", content=content.lstrip("\n").rstrip())
                )
            # 否则，将内容追加到前一条消息的 content 字段中
            else:
                messages[-1].content += content
                
         # 处理 user 角色
        elif role == "user":
            # 如果角色是 user，直接将消息内容添加到 messages 列表中
            messages.append(
                ChatMessage(role="user", content=content.lstrip("\n").rstrip())
            )
        else:
            # 如果角色不是 function、assistant 或 user，抛出错误
            raise ValueError(f"Incorrect role {role}.")
        
    query = _TEXT_COMPLETION_CMD
    # 如果最后一条消息的角色是 user
    if messages[-1].role == "user":
        # 这条消息的内容赋值给 query
        query = messages[-1].content
        # 移除最后一条 user 消息
        messages = messages[:-1]

    # messages 列表中的消息数量是否为偶数, 否则表示请求无效
    if len(messages) % 2 != 0:
        raise ValueError("length of message should be %2=0")

    # 初始化 history 列表，用于存储对话的历史记录，每条记录是一个包含用户提问和助手回答的二元组 [usr_msg, bot_msg]
    history = []
    # 以步长为2遍历 messages 列表, user和assistant的消息是配对的
    for i in range(0, len(messages), 2):
        # 检查当前消息对的角色是否符合预期，即当前消息是 usr_msg, 下一条消息是 bot_msg. 去除消息前后的换行符
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            # 如果存在系统消息且当前处理的是倒数第二条消息（用户消息），则将系统消息添加到用户消息之前，并将系统消息清空
            # 这确保在对话历史的最后一条用户消息之前包含系统消息
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            # 检查 bot_msg是否以某些预定义的思考标记(dummy_thought)
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t) :]
            history.append([usr_msg, bot_msg])
        else:
            raise ValueError("Expecting exactly one user (or function) role before every assistant role.")
    # 检查是否存在系统消息。如果 system 变量非空，则说明有系统消息(sys_msg) 需要处理
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nQuestion: {query}"
        
    # 返回两个值: query 包含最终的用户查询, history 包含格式化后的 usr_msg, bot_msg 的配对
    return query, history


'''
解析从模型返回的响应内容，并生成一个 ChatCompletionResponseChoice 对象
'''
def parse_response(response):
    func_name, func_args = "", ""
    # 字符串中最后出现 "xxx" 的位置
    i = response.rfind("\nAction:")
    j = response.rfind("\nAction Input:")
    k = response.rfind("\nObservation:")
    # 检查 "\nAction:" 是否在 "\nAction Input:" 之前
    # 如果是的话，就说明响应中包含了 Action 和 Action Input，但可能遗漏了 Observation
    # 如果确实遗漏了 Observation，则在响应的末尾添加 "\nObservation:"，以便后续处理
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
        # 重新找到 "\nObservation:" 的位置
        k = response.rfind("\nObservation:")
        # 从 response 中提取 func_name 和 func_args
        func_name = response[i + len("\nAction:") : j].strip()
        func_args = response[j + len("\nAction Input:") : k].strip()
    # 如果 func_name 非空
    if func_name:
        # 说明响应包含了函数调用信息。创建一个 ChatCompletionResponseChoice 对象
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=response[:i],
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
        return choice_data
    # 如果 response 中包含 "\nFinal Answer: "，则从该位置开始提取最终答案
    z = response.rfind("\nFinal Answer: ")
    if z >= 0:
        response = response[z + len("\nFinal Answer: ") :]
    # 创建一个 ChatCompletionResponseChoice 对象，其中 finish_reason 被设定为 "stop"，表示模型已完成生成响应
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return choice_data


# text_complete_last_message 函数用于生成对话历史中的最后一条消息的补全。它通过构建提示（prompt）从对话历史中生成补全，然后对生成的输出进行处理
def text_complete_last_message(history, stop_words_ids):
    # im_start 和 im_end 是标记对话中不同角色的特殊标记, 初始 prompt 设置为包含系统消息的文本
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    # 遍历 history 中的每个条目:
    for i, (query, response) in enumerate(history):
        # 去除查询和回应的前导和尾部 \n
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        # 将格式化的查询和回应附加到 prompt 中
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    # 从 prompt 中去除多余的结束标记，以确保格式正确
    prompt = prompt[: -len(im_end)]

    # 对结束标记 im_end 进行编码，并将其附加到 _stop_words_ids 列表中
    _stop_words_ids = [tokenizer.encode(im_end)]
    # 如果提供了stop_words_ids，它们也会被添加到 _stop_words_ids 中
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    # 将 prompt 编码(tokenizer)为 token ID
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    # 使用模型生成内容
    output = model.generate(input_ids, stop_words_ids=stop_words_ids).tolist()[0]
    # 将模型生成的输出解码回字符串
    output = tokenizer.decode(output, errors="ignore")
    
    # 输出处理
    # 确保生成的输出以 prompt 开始
    assert output.startswith(prompt)
    # 修剪输出中的多余标记
    output = output[len(prompt) :]
    output = trim_stop_words(output, ["<|endoftext|>", im_end])
    # 打印并返回
    print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
    return output
