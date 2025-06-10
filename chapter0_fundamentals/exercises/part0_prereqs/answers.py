import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"


# Einops


arr = np.load(section_dir / "numbers.npy")

print(arr.shape)

print(arr[0].shape)
display_array_as_img(arr[0])  # plotting the first image in the batch

print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])  # plotting the first channel of the first image, as monochrome

arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row


arr_stacked_v = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr_stacked_v)  # plotting all images, stacked in a column


arr_stacked_b = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1 = 2)
display_array_as_img(arr_stacked_b)  # plotting all images, stacked in a block

## Exercises - einops operations (match images)
### (1) Column stacking
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)  # plotting all images, stacked in a column

### (2) Column-stacking and copying
#### In this example we take just the first digit, and copy it along rows using einops.repeat.
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)

### (3) Row-stacking and double-copying
#### This example is pretty similar to the previous one, except that the part of the original image we need to slice 
#### and pass into einops.repeat also has a batch dimension of 2 (since it includes the first 2 digits).
arr3 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)

### (4) Stretching
#### The image below was stretched vertically by a factor of 2.
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)

### (5) Split channels
#### The image below was created by splitting out the 3 channels of the image (i.e. red, green, blue) 
#### and turning these into 3 stacked horizontal images. 
#### The output is 2D (the display function interprets this as a monochrome image).
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)

### (6) Stack into rows & cols
#### This requires a rearrange operation with dimensions for row and column stacking.
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)

### (7) Transpose
#### Here, we've just flipped the model's horizontal and vertical dimensions. 
#### Transposing is a fairly common tensor operation.
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)

### (8) Shrinking
#### Hint - for this one, you should use max pooling - i.e. 
#### each pixel value in the output is the maximum of the corresponding 2x2 square in the original image.
arr8 = einops.reduce(arr, "(b1 b2) c (h h1) (w w1) -> c (b1 h) (b2 w)", "max", b1=2, h1=2, w1=2)
display_array_as_img(arr8)