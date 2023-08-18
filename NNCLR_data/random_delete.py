# -*- coding: utf-8 -*-
"""
Program to randomly delete images in a folder until a given number of images
remain.

@author: manon
"""

import os
import random
import glob


def delete_random_images_until_count(folder_path, remaining_count):

    image_files = glob.glob(os.path.join(folder_path, "*.png"))

    while len(image_files) > remaining_count:

        # Randomly chose an image to delete
        image_to_delete = random.choice(image_files)
        
        # Delete image
        os.remove(image_to_delete)
        
        # Update list of images
        image_files = [file for file in image_files if file != image_to_delete]

if __name__ == "__main__":
    
    folder_path = "labelled/other"
    nb_deadwood_images = len(glob.glob(os.path.join('labelled/deadwood',
                                                    "*.png")))
    nb_remaining = int(2*nb_deadwood_images)
    
    delete_random_images_until_count(folder_path, nb_remaining)