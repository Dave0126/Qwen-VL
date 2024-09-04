import os
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import  FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field                           # 用于数据验证和解析
from sse_starlette.sse import EventSourceResponse               # 用于处理流式响应
from transformers import AutoTokenizer, AutoModelForCausalLM    # 加载模型和标记器
from transformers.generation import GenerationConfig

from utils import get_args
from openai_schema import ModelCard, ModelList, ChatMessage, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice
from openai_schema import ChatModelNotExists, ChatMessagesError, ChatFunctionCallNotAllow
from qwen_handle import _TEXT_COMPLETION_CMD, load_model, add_extra_stop_words, trim_stop_words, parse_messages, parse_response, text_complete_last_message

MODEL_NAME: Optional[str] = "Qwen/Qwen-VL-Chat-7B"
MODEL: Optional[AutoModelForCausalLM] = None
TOKENIZER: Optional[AutoTokenizer] = None


# 在 FastAPI 应用的生命周期结束时，清理 GPU 内存缓存。这对于长时间运行的服务很重要，可以避免内存泄漏
@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ChatModelNotExists)
async def chat_model_not_exists_exception_handler(request: Request, exc: ChatModelNotExists):
    """Handle the exception when the model does not exist."""
    print(request)
    return JSONResponse(status_code=404, content={
        "object": "error",
        "message": f"The model `{exc.model_name}` does not exist.",
        "type": "NotFoundError",
        "param": None,
        "code": 404
    })
    

@app.exception_handler(ChatMessagesError)
async def chat_messages_error_exception_handler(request: Request, exc: ChatMessagesError):
    """Handle the exception when the messages are not formatted correctly."""
    print(request)
    return JSONResponse(status_code=404, content={
        "object": "error",
        "message": f"The last message should be from the user.",
        "data": exc.messages,
        "type": "ValueError",
        "param": None,
        "code": 404
    })
    

@app.exception_handler(ChatFunctionCallNotAllow)
async def chat_function_call_not_allow_exception_handler(request: Request, exc: ChatFunctionCallNotAllow):
    """Handle the exception when the function call is not allowed."""
    print(request)
    return JSONResponse(status_code=404, content={
        "object": "error",
        "message": f"Function call `{exc.function_name}` is not allowed.",
        "type": "NotImplementedError",
        "param": None,
        "code": 404
    })
    
    
@app.get("/v1/models", response_model=ModelList, tags=["Models"])
async def list_models():
    global model_list
    return model_list


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest):
    print(f"Get request: {request}")
    global MODEL, TOKENIZER
    # verify model_name
    # if request.model != MODEL_NAME:
    #     raise ChatModelNotExists(model_name=request.model)

    try:
        # 如果请求中包含 functions，则确保 stop_words 列表包含 "Observation:"，用于处理模型生成的功能调用
        stop_words = add_extra_stop_words(request.stop)
        if request.functions:
            stop_words = stop_words or []
            if "Observation:" not in stop_words:
                stop_words.append("Observation:")

        # 解析消息和历史记录
        query, history = parse_messages(request.messages, request.functions)
        print(f"Get query: {query}, history: {history}")
    except ValueError as e:
        raise ChatMessagesError(messages=request.messages, exc=e.__str__())

    # functions and tools
    # if request.functions is not None or request.tools is not None:
    #     raise ChatFunctionCallNotAllow(function_name="")

    # # seed
    # if request.seed:
    #     torch.manual_seed(request.seed)

    # chat
    if request.stream:
        # todo: unSupported stream
        raise HTTPException(status_code=501, detail="Stream chat is not implemented.")
        # return StreamingResponse(stream_chat(query, history, MODEL, TOKENIZER, MODEL_NAME, append_history=False,
        #                          top_p=request.top_p, temperature=request.temperature))
    
    
    # 遍历 stop_words_ids 编码(tokenizer)为 token ID
    stop_words_ids = [TOKENIZER.encode(s) for s in stop_words] if stop_words else None
    # 生成响应
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(history, stop_words_ids=stop_words_ids)
    else:
        response, _ = MODEL.chat(
            TOKENIZER,
            query,
            history=history,
            stop_words_ids=stop_words_ids,
            append_history=False,
            top_p=request.top_p,
            temperature=request.temperature,
        )
        print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")
    # 处理和返回响应
    response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )
    
    
if __name__ == "__main__":
    args = get_args()
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"
        
    MODEL, TOKENIZER = load_model(args.checkpoint_path, trust_remote_code=True, device_map=device_map)
    model_list = ModelList()    
    model_list.data.append(ModelCard(id="Qwen-VL-Chat-7B"))
    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)