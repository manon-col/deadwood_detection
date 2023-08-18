# -*- coding: utf-8 -*-
"""
Program to flip images, with the aim of increasing the number of samples for
NNCLR training.

@author: manon-col
"""

import os
import numpy as np
from PIL import Image, ImageEnhance


def flip_images(folder):
    """
    Create up/down, left/right, and diagonal flipped images from images
    contained in a folder.

    Parameters
    ----------
    folder : string
        Where images are.

    """
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(folder) if f.endswith('.png')]

    for image_file in image_files:
        
        # Full path to input and output files
        input_path = os.path.join(folder, image_file)
        output_path_base = os.path.splitext(image_file)[0]

        image = Image.open(input_path)

        # Perform up/down flip
        # image_flip_up_down = image.transpose(Image.FLIP_TOP_BOTTOM)
        # output_path = os.path.join(folder,
        #                            f"{output_path_base}_flip_up_down.png")
        # image_flip_up_down.save(output_path)

        # Perform left/right flip
        image_flip_left_right = image.transpose(Image.FLIP_LEFT_RIGHT)
        output_path = os.path.join(folder,
                                    f"{output_path_base}_flip_left_right.png")
        image_flip_left_right.save(output_path)

        # Perform diagonal flip (up/down and left/right)
        # image_flip_diagonal = image_flip_up_down.transpose(Image.FLIP_LEFT_RIGHT)
        # output_path = os.path.join(folder,
        #                            f"{output_path_base}_flip_diagonal.png")
        # image_flip_diagonal.save(output_path)

def enhance_images(folder, factors):
    """
    Change image brightness.

    Parameters
    ----------
    folder : string
        Where images are.
    factors : list
        List of different factors of enhancement. (1=original, 0.5=dark,
        1.5=bright...).

    """
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(folder) if f.endswith('.png')]

    for image_file in image_files:
        
        # Full path to input and output files
        input_path = os.path.join(folder, image_file)
        output_path_base = os.path.splitext(image_file)[0]
        
        image = Image.open(input_path).convert("RGB")

        img_enhancer = ImageEnhance.Brightness(image)
        
        for factor in factors:
            
            enhanced_output = img_enhancer.enhance(factor)
            output_path = os.path.join(
                folder, f"{output_path_base}_enhance_{factor}.png")
            enhanced_output.save(output_path)

if __name__ == "__main__":
    
    # Here we don't have enough images in the "deadwood" class
    folder = 'labelled/deadwood'
    flip_images(folder)
    # enhance_images(folder, factors=[0.9,1.1])