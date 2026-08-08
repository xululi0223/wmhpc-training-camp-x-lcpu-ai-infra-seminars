"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def make_softmax(M, N, BLOCK_M=128, BLOCK_N=128, threads=128, dtype="float32"):
  @T.prim_func
  def main(
    X: T.Tensor((M, N), dtype),
    Y: T.Tensor((M, N), dtype),
  ):
    with T.Kernel(
      M,
      threads=threads,
    ) as bx:

      # 每个 block 处理一行
      X_frag = T.alloc_fragment((BLOCK_N,), dtype)

      block_max = T.alloc_fragment((1,), dtype)
      block_sum = T.alloc_fragment((1,), dtype)

      row_max = T.alloc_var(dtype)
      row_sum = T.alloc_var(dtype)

      row_max[0] = -T.infinity(dtype)

      for k in range(T.ceildiv(N, BLOCK_N)):
        for j in T.Parallel(BLOCK_N):
          gj = k * BLOCK_N + j

          if gj < N:
            X_frag[j] = X[bx, gj]
          else:
            X_frag[j] = -T.infinity(dtype)

        T.reduce_max(X_frag, block_max, dim=0)

        row_max[0] = T.max(row_max[0], block_max[0])

      row_sum[0] = 0.0

      for k in range(T.ceildiv(N, BLOCK_N)):
        for j in T.Parallel(BLOCK_N):
          gj = k * BLOCK_N + j
          if gj < N:
            X_frag[j] = T.exp(X[bx, gj] - row_max[0])
          else:
            X_frag[j] = 0.0
        
        T.reduce_sum(X_frag, block_sum, dim=0)
        row_sum[0] += block_sum[0]

      for k in range(T.ceildiv(N, BLOCK_N)):
        for j in T.Parallel(BLOCK_N):
          gj = k * BLOCK_N + j
          if gj < N:
            Y[bx, gj] = T.exp(X[bx, gj] - row_max[0]) / row_sum[0]
            
  return main

def softmax(x: torch.Tensor) -> torch.Tensor:
  kernel = tilelang.compile(make_softmax(x.shape[0], x.shape[1]))
  out = torch.empty_like(x)
  kernel(x, out)
  return out