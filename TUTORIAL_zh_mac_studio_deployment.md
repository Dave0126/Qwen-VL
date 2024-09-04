## 部署环境

```bash
Hardware Overview:
      Model Name: Mac Studio
      Chip: Apple M2 Ultra
      Total Number of Cores: 24 (16 performance and 8 efficiency)
      Memory: 192 GB
      System Firmware Version: 10151.81.1
      OS Loader Version: 10151.81.1
      
System Software Overview:
      System Version: macOS 14.3.1 (23D60)
      Kernel Version: Darwin 23.3.0
```

## 准备工作

建议使用多版本管理工具创建新的开发环境以将不同版本区分开来，如下以 `miniforge` 为例讲解。

要开始使用 `conda` 作为软件版本管理工具，您需要先安装一个基本的 Conda 环境。建议不要使用 Anaconda（可能存在商用授权等风险），可以使用由开源软件组织 [conda-forge](https://github.com/conda-forge) 开发维护的开源软件 [ `Miniforge`](https://github.com/conda-forge/miniforge)。具体安装步骤请参考 [GitHub 中的官方指南](https://github.com/conda-forge/miniforge#install)。

### 新建环境

以 MacOS 系统（类Unix）为例：可以使用 `curl` 或 `wget` 或其他程序下载安装程序，然后运行脚本

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
bash Miniforge3-MacOSX-$(uname -m).sh
```

或

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
bash Miniforge3-MacOSX-$(uname -m).sh
```

脚本执行结束后可以使用如下命令检查是否成功安装：

```bash
conda -V
# conda 24.5.0
```

在 `conda` 中创建 `Qwen-VL` 环境：

```bash
conda create --name qwen2-vl python=3.10 -y

# 在创建的环境中添加源(如有需要)
conda config --env --add channels conda-forge
conda config --env --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --env --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --env --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --env --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# 激活 qwen2-vl 环境
conda activate qwen2-vl
```

### 安装相关依赖

#### 安装 `Pytorch`

```bash
# 确保处于 qwen2-vl 环境
conda activate qwen2-vl

pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 测试 pytorch 是否安装成功
python -c "
import torch
import math                    
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
"
# True
# True
# ver. 2.4.0
```



#### 安装 `Tokenizers(huggingface)`

##### 前置依赖条件：安装 `RUST` 语言环境

```bash
# Tokenizers 的实现部分依赖于 RUST 语言，所以先安装 RUST
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh		# 官方方法
# 若下载速度太慢，可以添加国内镜像源
echo "export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup" >> ~/.zshrc
echo "export RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup" >> ~/.zshrc
source ~/.zshrc
# 检查是否成功安装
rustc --version
cargo --version
```

##### 安装 `Tokenizers`

```bash
# 使用 Github 仓库中的代码安装 Tokenizers
mkdir -p ~/Workspace/dependencies
cd ~/Workspace/dependencies
git clone https://github.com/huggingface/tokenizers
# 确保处于 qwen2-vl 环境
conda activate qwen2-vl
cd tokenizers
pip install setuptools_rust
python setup.py install
# ver. 0.19.1
```



#### 安装 `Transformer(huggingface)`

```bash
cd ~/Workspace/dependencies
# 使用 Github 仓库中的代码安装 Tokenizers
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
# 测试
python -c "
from transformers import pipeline;
print(pipeline('sentiment-analysis')('test for pipeline'))
"
# ver. 4.45.0
```

## 安装部署

### 准备 `Qwen-VL` 安装

```bash
# Clone Qwen-VL repos
mkdir -p ~/Workspace
cd ~/Workspace
git clone https://github.com/QwenLM/Qwen-VL.git
cd Qwen-VL

# 根据官方文档，使用不同的 requirement 文件安装依赖

# 基础框架调用
pip install -r requirements.txt

# WebAppDemo 演示
pip install -r requirements_web_demo.txt
python web_demo_mm.py
# 自定义配置
python web_demo_mm.py --checkpoint <YOUR_MODEL_DIR> --cpu-only --server-name 0.0.0.0

# OpenAI API 演示
pip install -r requirements_openai_api.txt
python openai_api.py
# 自定义配置
python openai_api.py --checkpoint <YOUR_MODEL_DIR> --cpu-only --server-name 0.0.0.0

```

> 注意：
>
> 此时程序会在默认 `cache` 中查找是否有缓存过的模型（`HuggingFace` 默认缓存地址在 `${HOME}/.cache/huggingface/hub/`），如果没有则会从网上下载。由于网络限制，程序会因为网络连接超时而中止。



### 下载 `Qwen-VL` 模型

一个 `Transformer` 类型的模型文件（[实例化大模型](https://huggingface.co/docs/transformers/v4.36.1/zh/big_models)），其目录下通常会有如下文件：

| 文件类型 | 文件说明                                                     | 备注                           |
| -------- | ------------------------------------------------------------ | ------------------------------ |
| 模型文件 | 预训练文件本体，文件体积过大时一般会分片为若干部分，`.bin`、`.ckpt`、`.safetensors` 等后缀名 | `model_pytorch.bin`            |
| 索引文件 | 大模型实例化时，根据索引文件，使得每个 `checkpoint` 的分片在前一个分片之后加载 | `pytorch_model.bin.index.json` |
| 分词文件 | 定义文本预处理过程中的分词规则和特殊字符处理方式             | `tokenizer.json`               |
| 模型配置 | 用于配置和调整模型的训练、评估和推理过程。通常包括 `config.json` 和 `finetune.json` 等文件 | `config.json`                  |
| 单词表   | 分词 `tokenizer` 过后产生的单词表文件                        | `vocab.json`                   |

下面给出两个开源模型分享平台：

- [`HuggingFace`](https://huggingface.co/)：国外开源模型分享社区
- [`ModelScope`（魔搭）](https://www.modelscope.cn/home)：国内开源模型分享社区，[模型下载教程（英文）](https://huggingface.co/docs/huggingface_hub/v0.24.6/guides/download)

可以在上述两个平台下载对应训练工具打包好的下载配置、词典、预训练模型（配置文件建议全部下载）等：

- `Pytorch` 对应模型：

  - `xxx.bin`（通常）

  - `xxx.safetensors`：需要安装 `safetensors` 库（安全存储和加载库）。特别针对 `Pytorch` 模型，通过加密和验证模型数据来增强安全性，防止模型数据被篡改。

    ```bash
    # 安装
    pip install safetensors
    ```

    ```python
    '''
    保存模型权重
    '''
    
    import torch
    import safetensors.torch import save_file
    
    # 假设 model 是你的模型实例
    model_state_dict = model.state_dict()
    # 使用 safetensors 保存
    save_file(model_state_dict, "model.safetensors")
    ```

    ```python
    '''
    加载模型权重
    '''
    
    import torch
    import safetensors.torch import load_file
    
    # 使用 safetensors 加载模型
    loaded_model_state_dict = load_file("model.safetensors")
    ```

- `TensorFlow` 对应模型：

  - `xxx.ckpt`

`Transformer` 框架提供一个将 `Tensorflow` 转换为 `PyTorch` 的指令：

```bash
transformer-cli convert \
--model_type <如bert等>\
--tf_checkpoint <原Tensorflow的ckpt文件地址>\
--cibfug <配置json文件地址>\
--pytorch_dump_output <目标PyTorch的bin文件地址>
```



## 自行实现的 OpenAI API 调用

```bash
curl -X GET http://localhost:8000/v1/models 


curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen-VL-Chat-7B",
  "messages": [
    {
      "role": "user",
      "content": [
            { "type": "text", "text":"这是哪里？"},
            { "type": "image", "image":"https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}
      	]
    },
    {
      "role": "assistant",
      "content": [
            { "type": "text", "text":"这是在海滩上，一个年轻女人和她的狗在沙滩上"}
      	]
    },
    {
      "role": "user",
      "content": [
            { "type": "text", "text":"是什么品种的狗？"}
      	]
    }
  ],
  "functions": [],
  "stop": [],
  "top_p": 0.9,
  "temperature": 0.7,
  "stream": false
}'

```

