#!/usr/bin/env python3


import click
import os
import re
# import requests
from pprint import pprint as pp
from pathlib import Path

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


IMAGE_TYPES = ['.jpg', '.png', '.gif']
VERBOTEN = '''.'"'''


def info(msg):
    click.secho(msg, fg='blue')


def caption_images(images):
    info('Model loading')
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    names = []
    for image in images:
        with Image.open(image) as img:
            info(f'Processing {image}')
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(
                pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0]

            names.append(generated_caption)
    return names


def get_command(caption, image):
    new_name = caption.replace(' ', '-').lower()
    new_name = ''.join([i for i in new_name if i not in VERBOTEN])
    new_name = re.sub('[-]+', '-', new_name)
    cmd = f'mv {image} {image.parent}/{new_name}{image.suffix}'
    return cmd


def print_command(cmd):
    click.secho(cmd, fg='yellow')


def run_command(cmd):
    click.secho('would rename file')


def process_images(images, for_real):
    if not images:
        click.secho('No images found')
        exit()

    captions = caption_images(images)

    for image, caption in zip(images, captions):
        command = get_command(caption, image)
        if for_real:
            print_command(command)
            run_command(command)
        else:
            print_command(command)


@click.group()
def caption():
    """Use ai to caption images."""
    pass


@caption.command()
@click.argument(
    'image_dir',
    type=click.Path(exists=True), # , file_ok=False<
    required=True,
)
@click.option('-r', '--for-real', is_flag=True)
def dir(image_dir, for_real):
    """Apply to a dir of images."""

    images = [
        image_dir / Path(i) for i in os.listdir(image_dir)
        if os.path.splitext(i)[1]
        in IMAGE_TYPES
    ]
    process_images(images, for_real)


@caption.command()
@click.argument("images", nargs=-1, required=True)
@click.option('-r', '--for-real', is_flag=True)
def images(images, for_real):
    """Apply to a list of images."""

    images = [Path(i) for i in images if os.path.splitext(i)[1] in IMAGE_TYPES]
    process_images(images, for_real)


if __name__ == "__main__":
    caption()
