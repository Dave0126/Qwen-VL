# OpenAI compatibility

> **Note:** OpenAI 兼容性处于实验阶段，可能会进行重大调整，包括重大更改。提供与部分 [OpenAI API](https://platform.openai.com/docs/api-reference) 的兼容性，以帮助将现有应用程序连接到模型服务。


## Usage

### OpenAI Python 库

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8000/v1/',

    # required but ignored
    api_key='IGNORED',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='Qwen-VL-Chat-7B',
)

response = client.chat.completions.create(
    model="Qwen-VL-Chat-7B",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://img3.chinadaily.com.cn/images/202111/03/61822e54a3107be47f279dc3.png",
                },
            ],
        }
    ],
    max_tokens=300,
)

completion = client.completions.create(
    model="Qwen-VL-Chat-7B",
    prompt="Say this is a test",
)

list_completion = client.models.list()

# model = client.models.retrieve("Qwen-VL-Chat-7B")

# embeddings = client.embeddings.create(
#     model="all-minilm",
#     input=["why is the sky blue?", "why is the grass green?"],
# )
```

### OpenAI JavaScript library

```javascript
import OpenAI from 'openai'

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1/',

  // required but ignored
  apiKey: 'IGNORED',
})

const chatCompletion = await openai.chat.completions.create({
    messages: [{ role: 'user', content: 'Say this is a test' }],
    model: 'Qwen-VL-Chat-7B',
})

const response = await openai.chat.completions.create({
    model: "Qwen-VL-Chat-7B",
    messages: [
        {
        role: "user",
        content: [
            { type: "text", text: "What's in this image?" },
            {
            type: "image_url",
            image_url: "https://img3.chinadaily.com.cn/images/202111/03/61822e54a3107be47f279dc3.png",
            },
        ],
        },
    ],
})

const completion = await openai.completions.create({
    model: "Qwen-VL-Chat-7B",
    prompt: "Say this is a test.",
})

const listCompletion = await openai.models.list()

// const model = await openai.models.retrieve("llama3")

// const embedding = await openai.embeddings.create({
//   model: "all-minilm",
//   input: ["why is the sky blue?", "why is the grass green?"],
// })
```

### `curl`

``` shell
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen-VL-Chat-7B",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "functions": [],
        "stop": [],
        "top_p": 0.9,
        "temperature": 0.7,
        "stream": false
    }'

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen-VL-Chat-7B",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "请描述一下这张图片?"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": "https://img3.chinadaily.com.cn/images/202111/03/61822e54a3107be47f279dc3.png"
                }
            }
            ]
        }
        ],
        "functions": [],
        "stop": [],
        "top_p": 0.9,
        "temperature": 0.7,
        "stream": false
  }'

curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen-VL-Chat-7B",
        "prompt": "Say this is a test"
    }'

curl -X GET http://localhost:8000/v1/models

# curl -X GET http://localhost:11434/v1/models/llama3

# curl -X GET http://localhost:11434/v1/embeddings \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "all-minilm",
#         "input": ["why is the sky blue?", "why is the grass green?"]
#     }'
```

## Endpoints

### `/v1/chat/completions`

#### Supported features

- [x] Chat completions
- [ ] Streaming
- [x] JSON mode
- [x] Reproducible outputs
- [x] Vision
- [x] Tools (streaming support coming soon)
- [ ] Logprobs

#### Supported request fields

- [x] `model`
- [x] `messages`
  - [x] Text `content`
  - [x] Image `content`
    - [x] Base64 encoded image
    - [x] Image URL
  - [x] Array of `content` parts
- [x] `frequency_penalty`
- [x] `presence_penalty`
- [x] `response_format`
- [ ] `seed`
- [ ] `stop`
- [ ] `stream`
- [ ] `temperature`
- [ ] `top_p`
- [ ] `max_tokens`
- [ ] `tools`
- [ ] `tool_choice`
- [ ] `logit_bias`
- [ ] `user`
- [ ] `n`

### `/v1/completions`

#### Supported features

- [ ] Completions
- [ ] Streaming
- [ ] JSON mode
- [ ] Reproducible outputs
- [ ] Logprobs

#### Supported request fields

- [ ] `model`
- [ ] `prompt`
- [ ] `frequency_penalty`
- [ ] `presence_penalty`
- [ ] `seed`
- [ ] `stop`
- [ ] `stream`
- [ ] `temperature`
- [ ] `top_p`
- [ ] `max_tokens`
- [ ] `suffix`
- [ ] `best_of`
- [ ] `echo`
- [ ] `logit_bias`
- [ ] `user`
- [ ] `n`

#### Notes

- `prompt` currently only accepts a string

### `/v1/models`

#### Notes

- `created` corresponds to when the model was last modified
- `owned_by` corresponds to the ollama username, defaulting to `"library"`

### `/v1/models/{model}`

#### Notes

- `created` corresponds to when the model was last modified
- `owned_by` corresponds to the ollama username, defaulting to `"library"`

### `/v1/embeddings`

#### Supported request fields

- [ ] `model`
- [ ] `input`
  - [ ] string
  - [ ] array of strings
  - [ ] array of tokens
  - [ ] array of token arrays
- [ ] `encoding format`
- [ ] `dimensions`
- [ ] `user`
