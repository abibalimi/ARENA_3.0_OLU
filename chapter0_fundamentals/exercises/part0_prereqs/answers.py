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
## Difficulty: 🔴🔴🔴⚪⚪
## Importance: 🔵🔵⚪⚪⚪
## You should spend up to ~45 minutes on these exercises collectively.
## If you think you get the general idea, then you can skip to the next section.
## You shouldn't spend longer than ~10 mins per exercise.

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



## Exercises - einops operations & broadcasting
## Difficulty: 🔴🔴⚪⚪⚪
## Importance: 🔵🔵🔵⚪⚪
## You should spend up to ~45 minutes on these exercises collectively.
## These are more representative of the kinds of einops operations you'll use in practice.

def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")
    
### (A1) rearrange
#### We'll start with a simple rearrange operation -
#### you're asked to return a particular tensor using only t.arange and einops.rearrange.
#### The t.arange function is similar to the numpy equivalent:
#### torch.arange(start, end) will return a 1D tensor containing all the values from start to end - 1 inclusive.
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    tsr = t.arange(3, 9)
    return einops.rearrange(tsr, "(a b) -> a b", a=3, b=2)

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


### (A2) rearrange
#### This exercise has the same pattern as the previous one.
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1, 7), "(h w) -> h w", h=2, w=3)

assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))



### (B1) temperature average
#### Here you're given a 1D tensor containing temperatures for each day.
#### You should return a 1D tensor containing the average temperature for each week.
#### This could be done in 2 separate operations (a reshape from 1D to 2D followed by taking the mean over one of the axes),
#### however we encourage you to try and find a solution with einops.reduce in just a single line.
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(w 7) -> w","mean")

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))


### (B2) temperature difference
#### Here, we're asking you to subtract the average temperature from each week from the daily temperatures.
#### You'll have to be careful of broadcasting here, since your temperatures tensor has shape (14,)
#### while your average temperature computed above has shape (2,) - these are not broadcastable.
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg = temperatures_average(temps)  # avg = einops.reduce(temps, "(w 7) -> w","mean")
    return temps - einops.repeat(avg, "w -> (w 7)")

expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))


### (B3) temperature normalized
#### Lastly, you need to subtract the average and divide by the standard deviation. 
#### Note that you can pass t.std into the einops.reduce function to return the std dev of the values you're reducing over.
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    avg = einops.reduce(temps, "(w 7) -> w", "mean")
    std = einops.reduce(temps, "(w 7) -> w", t.std)
    diff = temps - einops.repeat(avg, "w -> (w 7)")
    return diff / einops.repeat(std, "w -> (w 7)")

expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))


### (C1) normalize a matrix
#### Here, we're asking you to normalize the rows of a matrix so that each row has L2 norm (sum of squared values) equal to 1.
#### Note - L2 norm and standard deviation are not the same thing; L2 norm leaves out the averaging over size of vector step.
#### We recommend you try and use the torch function t.norm directly rather than einops for this task.
#### Two useful things we should highlight here:
#### Most PyTorch functions like t.norm(tensor, ...) which operate on a single tensor are also tensor methods, i.e.
#### they can be used as tensor.norm(...) with the same arguments.
#### Most PyTorch dimensionality-reducing functions have the keepdim argument,
#### which is false by default but will cause your output tensor to keep dummy dimensions if set to true.
#### For example, if tensor has shape (3, 4) then tensor.sum(dim=1) has shape (3,) but tensor.sum(dim=1, keepdim=True) has shape (3, 1).
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    row_norms = matrix.norm(dim=1, keepdim=True)
    return matrix / row_norms

matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)


### (C2) pairwise cosine similarity
#### Now, you should compute a matrix of shape (m, m)
#### where out[i, j] is the cosine similarity between the i-th and j-th rows of matrix.
#### The cosine similarity between two vectors is given by summing the elementwise products of the normalized vectors.
#### We haven't covered einsum yet, but you should be able to get this working using normal elementwise multiplication
#### and summing (or even do this in one step - can you see how?).
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.
    The reason this solution works is that matrix_normalized @ matrix_normalized.T multiplies along the columns of matrix_normalized 
    and the rows of matrix_normalized.T then sums the output - in other words, it computes the dot products!

    matrix: shape (m, n)
    """
    normalized_matrix = normalize_rows(matrix)
    return normalized_matrix @ normalized_matrix.T

matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)


### (D) sample distribution
#### Here we're having you do something a bit more practical and less artificial. 
#### You're given a probability distribution (i.e. a tensor of probabilities that sum to 1) and asked to sample from it.
#### Hint - you can use the torch functions t.rand and t.cumsum to do this without any explicit loops.
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.
    
    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples
    
    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    probs_cumsum = probs.cumsum(dim=0)
    samples = t.rand(n, 1)
    return (samples > probs_cumsum).sum(dim=-1)

n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)



### (E) classifier accuracy
#### Here, we're asking you to compute the accuracy of a classifier. scores is a tensor of shape (batch, n_classes)
#### where scores[b, i] is the score the classifier gave to class i for input b, and true_classes is a tensor of shape (batch,)
#### where true_classes[b] is the true class for input b.
#### We want you to return the fraction of times the maximum score is equal to the true class.
#### You can use the torch function t.argmax, it works as follows:
#### tensor.argmax(dim) will return a tensor of the index containing the maximum value along the dimension dim
#### (i.e. the shape of this output will be the same as the shape of tensor except for the dimension dim).
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    max_idx = t.argmax(scores, dim=1)
    return (max_idx == true_classes).sum() / len(true_classes)  # (scores.argmax(dim=1) == true_classes).float().mean()

scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected
print("Tests passed!")



### (F1) total price indexing
#### The next few exercises involve indexing, often using the torch.gather function. You can read about it in the docs.
#### If you find gather confusing, an alternative is the eindex library, which was designed to provide indexing features motivated by how einops works.
#### You can read more about that here, and as a suggested bonus you can try implementing / rewriting the next few functions using eindex.
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    return prices[items].sum().item()

prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0
print("Tests passed!")



### (F2) gather 2D
def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output.
    This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # YOUR CODE HERE - add assert statement(s) here for `indices` and `matrix`
    assert matrix.ndim == indexes.ndim, f"Dimension mismatch, got: {matrix.ndim = } and {indexes.ndim = }"
    assert indexes.shape[0] <= matrix.shape[0], f"Shape mismatch: {indexes.shape[0] = } must be <= to {matrix.shape[0] = }"
    

    out = matrix.gather(1, indexes)
    # YOUR CODE HERE - add assert statement(s) here for `out`
    assert out.shape == indexes.shape, f"Shape mismatch, got: {out.shape = } and {indexes.shape = }"

    return out

matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)

indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)



### (F3) total price gather
def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather. """
    assert items.max() < prices.shape[0]
    return prices.gather(0,items).sum().item()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0
print("Tests passed!")



### (G) indexing
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    return matrix[tuple(coords.T)] # matrix[tensor([0, 0, 1]), tensor([1, 4, 4])]


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))

mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))



### (H1) batched logsumexp
def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    maxes = matrix.max(dim=-1).values
    diff = matrix - einops.rearrange(maxes, "n -> n 1") # maxes[:, None] or .unsqueeze(1), broadcasting
    expo = t.exp(diff)
    return maxes + t.log(t.sum(expo, dim=-1))

matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)

matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)



### (H2) batched softmax
def batched_softmax(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    expos = matrix.exp()
    return expos / expos.sum(dim=-1, keepdim=True)


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))  # check it's translation-invariant

matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))



### (H3) batched logsoftmax
def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n).

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    maxes = matrix.max(dim=-1, keepdim=True).values
    # log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
    X = matrix - maxes # subtract the maximum first
    return X - X.exp().sum(dim=-1, keepdim=True).log()

matrix = t.arange(1, 7).view((2, 3)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 7).view((2, 3)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-2.4076, -1.4076, -0.4076],
                     [-2.4076, -1.4076, -0.4076]])
assert_all_close(actual, expected)