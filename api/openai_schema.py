import json
import time
from uuid import uuid4
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field                           # 用于数据验证和解析

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []
    

class ImageMessageContent(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[str] = None    


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[
        Union[str, List[ImageMessageContent]]
        ] = None
    function_call: Optional[Dict] = None
    

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = Field(default=None, ge=0.1, le=2)
    top_p: Optional[float] = Field(default=None, ge=0.1, le=1)
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"] = Field(default="chat.completion")
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    

class ChatModelNotExists(Exception):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        
class ChatMessagesError(Exception):
    def __init__(self, messages: List[ChatMessage], exc: str = None):
        self.messages = messages
        self.exc = exc
        
        
class ChatFunctionCallNotAllow(Exception):
    def __init__(self, function_name: str = None):
        self.function_name = function_name