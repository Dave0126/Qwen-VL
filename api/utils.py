import os
import re
from uuid import uuid4
from argparse import ArgumentParser
from base64 import b64decode, b64encode

from requests import get as requests_get

FILE_CACHE_DIR = os.path.expanduser("~/.cache/image/temp")  # 缓存目录路径


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="QWen/QWen-7B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--apple-silicon-only", action="store_true", help="Run demo with Apple Silicon GPU by mps"
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
        " If you want other computers to access your server, use 0.0.0.0 instead.",
    )
    args = parser.parse_args()
    return args


def check_file_exists(file_id: str):
    """检查文件是否存在"""
    if not os.path.exists(os.path.join(FILE_CACHE_DIR, file_id)):
        raise ValueError(f"FileNotFound: {file_id}")
    

def download_img_from_url(url: str, save_dir: str = FILE_CACHE_DIR if FILE_CACHE_DIR else "") -> str:
    """
    Download the image from the url.

    :param url: The image url.
    :param save_dir: The image save directory.
    :return: The image save path.
    """
    match = re.match(r'^data:(?P<mime_type>image/.+);base64,(?P<base64_data>.+)', url)  # 匹配 Base64 编码数据的常见前缀
    if match:
        # base64 data
        img_data = b64decode(match.group('base64_data'))
        extension = match.group('mime_type').split('/')[1]
    elif '127.0.0.1' in url or 'localhost' in url:
        # todo: 待优化本地匹配逻辑
        # 针对 autodl 的 'seetacloud.com' 进行特殊处理
        # 解析url中的路径部分，匹配file_id
        file_id = url.split('/')[-2]
        check_file_exists(file_id)
        return os.path.join(FILE_CACHE_DIR, file_id)
    else:
        # url
        response = requests_get(url)
        print(response.headers['content-type'].split('/')[0])
        if (response.headers['content-type'].split('/')[0] != 'image'):
            raise ValueError(f"FileError: URL {url} is not an image")
        img_data = response.content
        extension = response.headers['content-type'].split('/')[1].split(';')[0]
        print(f"Download Image, url: {url}, extension: {extension}")
    # save image
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"image_{uuid4().hex[:8]}.{extension}")
    with open(img_path, 'wb') as f:
        f.write(img_data)
    return img_path


def img_to_base64(img_path: str) -> str:
    """Convert the image to base64."""
    with open(img_path, 'rb') as f:
        return b64encode(f.read()).decode()
    

if __name__ == "__main__":
    dir = download_img_from_url(
        url = "https://img3.chinadaily.com.cn/images/202111/03/61822e54a3107be47f279dc3.png",
        save_dir = FILE_CACHE_DIR
    )
    base64 = img_to_base64(dir)
    print(base64)