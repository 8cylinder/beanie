#!/usr/bin/env python3


import click
import os
import requests
from pprint import pprint as pp

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

images_dir = 'images'
images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
for image in images:
    with Image.open(image) as img:

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        click.secho(image, bold=True)
        click.secho(f'    {generated_caption}')


# import os
# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from pprint import pprint as pp
#
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
#
# images_dir = 'images'
# images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
#
# for image in images:
#     with Image.open(image) as img:
#         # print(f"Processing image {image} with size {img.size}")
#         # pp(img)
#         # continue
#
#         # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
#         # image = Image.open(img).convert('RGB')
#         # pp(image)
#         # continue
#
#         # conditional image captioning
#         inputs = processor(img, return_tensors="pt")
#         out = model.generate(**inputs)
#         print(image)
#         print('   ', processor.decode(out[0], skip_special_tokens=True))
#
#         # # unconditional image captioning
#         # inputs = processor(image, return_tensors="pt")
#         # out = model.generate(**inputs)
#         # print(processor.decode(out[0], skip_special_tokens=True))




# import click
# from transformers import ViTImageProcessor
# from transformers import VisionEncoderDecoderModel
# from transformers import AutoTokenizer
# import torch
# from PIL import Image
# from pprint import pprint as pp
#
#
# def predict_step(image_paths):
#     model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#
#     # max_length = 16
#     # num_beams = 4
#     max_length = 50
#     num_beams = 4
#     gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
#
#     images = []
#     for image_path in image_paths:
#         i_image = Image.open(image_path)
#         if i_image.mode != "RGB":
#             i_image = i_image.convert(mode="RGB")
#
#         images.append(i_image)
#
#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]
#     return preds
#
#
# @click.command()
# @click.argument("image_paths", nargs=-1, required=True)
# def caption_images(image_paths):
#     captions = predict_step(image_paths)
#     for caption, image in zip(captions, image_paths):
#         print(image)
#         print(f'    {caption}')
#
#
# if __name__ == "__main__":
#     caption_images()
