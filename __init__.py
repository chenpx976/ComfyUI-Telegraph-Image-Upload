
from io import BytesIO
import numpy as np

from PIL import Image, ImageOps
import torch
import requests
import io


# def tensor2pil(image):
#     return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# 通过 URL 获取图片
def get_image_from_url(url):
    response = requests.get(url, timeout=5)
    if response.status_code != 200:
        raise Exception(response.text)

    i = Image.open(io.BytesIO(response.content))

    i = ImageOps.exif_transpose(i)

    if i.mode != "RGBA":
        i = i.convert("RGBA")

    # recreate image to fix weird RGB image
    alpha = i.split()[-1]
    image = Image.new("RGB", i.size, (0, 0, 0))
    image.paste(i, mask=alpha)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    return image, mask


def restore_image(image_tensor, mask=None):
    # 假设 image_tensor 是一个 [1, C, H, W] 形状的张量
    # 将 PyTorch tensor 转换回 numpy 数组
    image = image_tensor.squeeze().numpy()  # 移除批处理的维度

    # 将值范围从 [0, 1] 还原回 [0, 255] 并转换为 uint8
    image = (image * 255.0).astype(np.uint8)

    # 如果提供了 mask，则还原遮罩并将其应用于图像
    if mask is not None:
        # 将遮罩值范围从 [0, 1] 还原回 [0, 255]
        mask = (mask * 255.0).astype(np.uint8)
        # 假设遮罩是一个 H x W 的数组，我们需要将其扩展为 H x W x 1 以便与图像合并
        mask = np.expand_dims(mask, axis=-1)
        # 将遮罩添加到图像数据中作为第四个通道（RGBA）
        image = np.concatenate([image, mask], axis=-1)

    # 将 numpy 数组转换回 PIL 图像
    if image.shape[2] == 3:
        restored_image = Image.fromarray(image, 'RGB')
    else:
        restored_image = Image.fromarray(image, 'RGBA')

    return restored_image


def image_to_buffer(image):
    """
    将 PIL 图像对象转换为二进制缓冲区

    参数:
    image -- PIL 图像对象

    返回:
    一个包含图像数据的 BytesIO 对象
    """
    buffer = BytesIO()
    image.save(buffer, format=image.format if image.format else 'PNG')  # 保存图像到 buffer 中，格式与原图像相同
    buffer.seek(0)  # 将读写位置移动到缓冲区的开头
    return buffer


class TelegraphImageUpload_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "url": ("STRING", {"default": "https://im.gurl.eu.org"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("url", "image")
    CATEGORY = "TelegraphImageUpload_Node"
    FUNCTION = "execute"

    def execute(self, image, url):
        print('Uploading image', image)
        buffer = image_to_buffer(restore_image(image))
        print('buffer:', buffer)
        files = {'file': ('image.png', buffer, 'image/png')}  # 'file'是你的接收服务器端的文件字段
        response = requests.post(
            url=url + '/upload',
            headers={
                "Accept": "application/json, text/plain, */*",
            },
            files=files
        )
        if response.status_code != 200:
            # 更详细的错误信息
            error_info = response.json().get('message', 'Unknown Error')
            raise Exception(f'Error: {error_info}')
        print(f'Upload response: {response.json()}')
        url = url + response.json()[0]["src"]
        # 使用logging代替print进行调试
        print(f'Uploading file to {url}')
        preview = get_image_from_url(url)
        return (url, preview[0])


# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "TelegraphImageUpload_Node": TelegraphImageUpload_Node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TelegraphImageUpload_Node": "TelegraphImageUpload",
}
