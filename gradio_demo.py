import json

import torch

import gradio as gr
import pandas as pd
from skimage import data
from PIL import Image
from torchkeras import plots
from torchkeras.data import get_url_img
from pathlib import Path

from models.models import MODELS

config = json.load(open("./configs/config_HSIT_PRE.json"))
model = MODELS['HSIT_PRE'](config)
checkpoint = torch.load("./Experiments/HSIT/LRHR_dataset/N_modules(4)/best_model.pth")
model.load_state_dict(checkpoint, strict=False)


# 会在网页上给出的几个示例图像，这里的格式得改一下
Image.fromarray(data.coffee()).save('1.jpeg')
Image.fromarray(data.astronaut()).save('2.jpeg')
Image.fromarray(data.cat()).save('3.jpeg')



def detect(img,img2):
    if isinstance(img, str): # 这里需要改成pansharpening的图像格式
        img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
    if isinstance(img2, str): # 这里需要改成pansharpening的图像格式
        img2 = get_url_img(img2) if img2.startswith('http') else Image.open(img2).convert('RGB')
    # result = model.predict(source=img)#模型结果
    print(img,img2)
    result = model(img, img2)
    return result


with gr.Blocks() as demo:
    gr.Markdown("# Pansharpening可视化测试")

    # with gr.Tab("捕捉摄像头"):
    #     in_img = gr.Image(source='webcam', type='pil')
    #     button = gr.Button("执行", variant="primary")
    #
    #     gr.Markdown("## 输出")
    #     out_img = gr.Image(type='pil')
    #
    #     button.click(detect,
    #                  inputs=in_img,
    #                  outputs=out_img)

    with gr.Tab("选择示例图片"):
        files = ['1.jpeg', '2.jpeg', '3.jpeg']
        drop_down = gr.Dropdown(choices=files, value=files[0])
        button = gr.Button("执行", variant="primary")
        drop_down2 = gr.Dropdown(choices=files, value=files[1])

        gr.Markdown("## 输出")
        out_img = gr.Image(type='pil')

        button.click(detect,
                     inputs=[drop_down,drop_down2],
                     outputs=out_img)

    with gr.Tab("输入图片链接"):
        default_url = ''#网页链接
        url = gr.Textbox(value=default_url)
        url2 = gr.Textbox(value=default_url)
        button = gr.Button("执行", variant="primary")

        gr.Markdown("## 输出")
        out_img = gr.Image(type='pil')

        button.click(detect,
                     inputs=[url,url2],
                     outputs=out_img)

    with gr.Tab("上传本地图片"):
        input_img = gr.Image(type='pil')
        input_img2 = gr.Image(type='pil')
        button = gr.Button("执行检测", variant="primary")

        gr.Markdown("## 预测输出")
        out_img = gr.Image(type='pil')

        button.click(detect,
                     inputs=[input_img,input_img2],
                     outputs=out_img)

gr.close_all()
demo.queue(concurrency_count=3)
demo.launch(share=True)