import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
from plotly_utils import imshow

MAIN = __name__ == "__main__"

"""Today we'll be practicing batched matrix operations in PyTorch by writing a basic graphics renderer. 
We'll start with an extremely simplified case and work up to rendering your very own 3D Pikachu!
We'll also be touching on some general topics which will be important going forwards in this course, such as:
Using GPT systems to assist your learning and coding
Typechecking, and good coding practices
Debugging, with VSCode's built-in run & debug features"""



# 1️⃣ Rays & Segments
