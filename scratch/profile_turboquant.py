import cProfile
import pstats
import torch
import sys
import os

# Đường dẫn tới thư mục dự án
sys.path.append(os.getcwd())

from scripts.benchmark_e2e import main

def profile_benchmark():
    # Chạy main của benchmark với cProfile
    # Giới hạn số token để profile nhanh
    # Lưu ý: Sửa scripts/benchmark_e2e.py tạm thời để chỉ chạy 5 tokens khi profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Giả lập chạy benchmark (chúng ta sẽ bắt exception nếu nó dừng quá sớm)
        main()
    except Exception as e:
        print(f"Benchmark output caught: {e}")
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30) # In ra 30 hàm tốn thời gian nhất

if __name__ == "__main__":
    profile_benchmark()
