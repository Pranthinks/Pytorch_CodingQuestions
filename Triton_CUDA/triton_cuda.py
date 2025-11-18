#Implementing Custom CUDA Kernals with triton
import triton
import triton.language as tl
import torch

device = 'cuda'if torch.cuda.is_available() else 'cpu'


#Function to add 1 to each element in a tensor
@triton.jit
def CUDA_Increment(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(0)
  offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offset < n_elements

  val = tl.load(x_ptr + offset, mask = mask)
  val = val + 1
  tl.store(x_ptr + offset, val, mask = mask)

'''
BLOCK_SIZE = 128
grid = lambda meta: (triton.cdiv(x.numel(), BLOCK_SIZE), )
CUDA_Increment[grid](x, x.numel(), BLOCK_SIZE = BLOCK_SIZE)

print(x)
'''
#Function to add two 1-D Tensors

x = torch.tensor([1.0, 2.0, 3.2, 4.5, 4.9, 6.3], dtype = torch.float32).to(device)
y = torch.tensor([6.0, 5.0, 6.2, 0.5], dtype = torch.float32).to(device)
out = torch.empty_like(x)
pids = torch.empty_like(x)

@triton.jit

def Add_Two_Tensors(x_ptr, y_ptr, out_ptr, out_pid, n_elements1, n_elements2,  BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(0)
  offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offset < max(n_elements1, n_elements2)
  x = tl.load(x_ptr + offset, mask = offset < n_elements1, other = 0.0)
  y = tl.load(y_ptr + offset, mask = offset < n_elements2, other = 0.0)
  out = x + y
  tl.store(out_ptr + offset, out, mask = mask)
  tl.store(out_pid + offset, pid, mask = mask)

BLOCK_SIZE = 2
grid = lambda meta: (triton.cdiv(max(x.numel(), y.numel()),  BLOCK_SIZE), )
Add_Two_Tensors[grid](x, y, out, pids , x.numel(), y.numel(), BLOCK_SIZE = BLOCK_SIZE)
print(out)
print(pids.cpu())


#Multiplying two numbers in 1-D Matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.arange(1, 10, 1, dtype = torch.float32).to(device)
y = torch.arange(2, 25, 2, dtype = torch.float32).to(device)
output = torch.empty_like(y)
pids_op = torch.empty_like(y)

@triton.jit

def Cuda_Multi(x_ptr, y_ptr, out_ptr, pids, len_x, len_y, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(0)

  offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offset < max(len_x, len_y)

  x = tl.load(x_ptr + offset , mask = offset < len_x, other = 1.0)
  y = tl.load(y_ptr + offset , mask = offset < len_y, other = 1.0)
  out = x * y
  tl.store(out_ptr + offset, out, mask = mask)
  tl.store(pids + offset, pid, mask = mask)

BLOCK_SIZE = 2
grid = lambda meta: (triton.cdiv(max(x.numel(), y.numel()),  BLOCK_SIZE), )
Cuda_Multi[grid](x, y, output, pids_op, x.numel(), y.numel(), BLOCK_SIZE = BLOCK_SIZE)
print(output)
print(pids_op.cpu())


#Writing Custom CUDA Kernals to do arthimatic operion on 2D Arrays

x = torch.ones(4, 4, dtype = torch.int).to(device)
y = torch.ones(2, 3, dtype = torch.int).to(device)
out = torch.empty_like(x)
padd = torch.empty_like(x)

@triton.jit
def Addition_2d(x_ptr, y_ptr, out_ptr, pad ,len_x, len_y, stride_x, stride_y, stride_out,BLOCK_SIZE :tl.constexpr):
  pid1 = tl.program_id(0)
  pid2 = tl.program_id(1)

  row_offset = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  col_offset = pid2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

  rows = row_offset[:, None]
  cols = col_offset[None, :]

  mask = (rows < len_x) & (cols < len_y)

  x = tl.load(x_ptr + rows * stride_x + cols , mask = mask, other = 0.0)
  y = tl.load(y_ptr + rows * stride_y + cols , mask = mask, other = 0.0)
  out = x + y

  tl.store(out_ptr + rows * stride_out + cols, out, mask = mask)
  tl.store(pad + rows * stride_out + cols, pid1, mask = mask)


BLOCK = 2
grid = (triton.cdiv(x.shape[0], BLOCK), triton.cdiv(x.shape[1], BLOCK))

Addition_2d[grid](x, y, out, padd, x.shape[0], x.shape[1], x.stride(0), y.stride(0), out.stride(0), BLOCK)
print(out)
print(padd)

import torch
import triton
import triton.language as tl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.ones(3, 3, dtype = torch.int).to(device)
y = torch.ones(3, 3, dtype = torch.int).to(device)

out = torch.empty((3, 3), dtype = torch.int).to(device)
pids = torch.empty((3, 3), dtype = torch.int).to(device)

@triton.jit
def Division_2D(x_ptr, y_ptr, out_ptr, pids_ptr, len_x, len_y, stride_x , stride_y, stride_out, BATCH_SIZE: tl.constexpr):
  pid0 = tl.program_id(0)
  pid1 = tl.program_id(1)

  row_offset = pid0 * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
  col_offset= pid1 * BATCH_SIZE + tl.arange(0, BATCH_SIZE)

  rows = row_offset[:, None]
  cols = col_offset[None, :]

  mask = (rows < len_x) & (cols < len_y)
  x = tl.load(x_ptr + rows * stride_x + cols, mask = mask, other = 1.0)
  y = tl.load(y_ptr + rows * stride_y + cols, mask = mask, other = 1.0)

  out = x / y

  tl.store(out_ptr + rows * stride_out + cols, out, mask = mask)
  tl.store(pids_ptr + rows * stride_out + cols, pid0, mask = mask)

BATCH_SIZE = 2
grid = (triton.cdiv(x.shape[0], BATCH_SIZE), triton.cdiv(x.shape[1], BATCH_SIZE))

Division_2D[grid](x, y, out, pids, x.shape[0], x.shape[1], x.stride(0), y.stride(0), out.stride(0), BATCH_SIZE )
print(out)
print(pids)


