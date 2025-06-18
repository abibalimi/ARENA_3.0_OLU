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
## 1D Image Rendering

### Exercise - implement make_rays_1d

### Difficulty: 🔴🔴⚪⚪⚪
### Importance: 🔵🔵🔵⚪⚪
### You should spend up to 10-15 minutes on this exercise.
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    tip: the 'out' keyword argument: If provided, instead of allocating a new tensor and returning that, the output is written directly to the out tensor.
    """
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(start=-y_limit, end=y_limit, steps=num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)



## Ray-Object Intersection

fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)

@interact(v=(0.0, 6.0, 0.01), seed=list(range(10)))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
        
### Exercise - implement intersect_ray_1d
### Difficulty: 🔴🔴🔴⚪⚪
### Importance: 🔵🔵🔵🔵⚪

### You should spend up to 20-25 minutes on this exercise.
### It involves some of today's core concepts - tensor manipulation, linear operations, etc.