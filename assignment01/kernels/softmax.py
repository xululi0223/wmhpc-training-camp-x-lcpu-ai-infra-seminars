"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, m, n, BLOCK_N: tl.constexpr):
  row = tl.program_id(0)

  cols = tl.arange(0, BLOCK_N)
  mask = cols < n

  x_ptrs = X + row * N + cols

  x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

  x_max = tl.max(x, axis=0)

  numerator = tl.exp(x - x_max)

  denominator = tl.sum(numerator, axis=0)

  y = numerator / denominator

  y_ptrs = Y + row * N + cols
  tl.store(y_ptrs, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
  M, N = x.shape
  BLOCK_N = triton.next_power_of_2(N)
  grid = (M,)

  y = torch.empty_like(x)

  softmax_kernel[grid](
    x, y, n=N, BLOCK_N=BLOCK_N
  )

  return y