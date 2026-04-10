import torch
import triton
import triton.language as tl

@triton.jit
def test_loop_kernel(N, MAX_BLOCKS, BLOCK_SIZE: tl.constexpr):
    b_idx = 0
    # Test while loop (SOTA for dynamic bounds)
    while (b_idx * BLOCK_SIZE) < N:
        # Dummy op
        b_idx += 1

def run_test():
    N = 100
    MAX_BLOCKS = 1024
    BLOCK_SIZE = 64
    test_loop_kernel[(1,)](N, MAX_BLOCKS, BLOCK_SIZE)
    print("Kernel compiled and ran successfully with while loop.")

if __name__ == "__main__":
    run_test()
