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