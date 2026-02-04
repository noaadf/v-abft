import torch
import torch_npu
import utils

n = 8192
trials = 1000
dtypes = [torch.bfloat16, torch.float32, torch.float16]
dtypes = [torch.float32]

# 定义四种矩阵初始化方式及其参数
init_methods = [
    {
        'name': 'generate_matrice_normal',
        'func': utils.generate_matrice,
        'params': {'std': 1, 'mean': 1}
    },
    {
        'name': 'generate_matrice_clamp',
        'func': utils.generate_matrice_clamp,
        'params': {'std': 1, 'mean': 1}
    },
    {
        'name': 'generate_matrice',
        'func': utils.generate_matrice,
        'params': {'std': 1, 'mean': 1e-6}
    },
    {
        'name': 'generate_matrice_uniform',
        'func': utils.generate_matrice_uniform,
        'params': {'lower': -1, 'upper': 1}
    }
]

output_file1 = "error_injection_results0.txt"
output_file2 = "non_error_test_results.txt"

with open(output_file1, "a") as f:
    f.write("Error Injection Test Results\n")
    f.write("=" * 50 + "\n\n")
    
    for init_method in init_methods:
        f.write(f"Initialization Method: {init_method['name']}\n")
        f.write(f"Parameters: {init_method['params']}\n")
        f.write("-" * 50 + "\n")
        
        for dtype in dtypes:
            f.write(f"\nData type: {dtype}\n")
            print(f"Initialization Method: {init_method['name']}")
            print(f"Data type: {dtype}")
            
            total_rate = 0
            valid_bit_count = 0
            
            exponential_bits = {
                torch.bfloat16: range(7, 16),
                torch.float16: range(10, 16),
                torch.float32: range(23, 32)
            }

            for bit in exponential_bits[dtype]:
                error_count = 0
                valid_count = 0
                max_attempts = trials * 5  # 最大尝试次数，防止无限循环
                attempts = 0
                
                # 尝试注入错误直到达到目标成功次数或超过最大尝试次数
                while valid_count < trials and attempts < max_attempts:
                    attempts += 1
                    
                    # 使用指定的初始化方法生成矩阵
                    A = init_method['func'](
                        128, n, 
                        device=utils.device_npu, 
                        dtype=dtype,
                        **init_method['params']
                    )
                    B = init_method['func'](
                        n, 256,
                        device=utils.device_npu, 
                        dtype=dtype,
                        **init_method['params']
                    )
                    if(dtype == torch.float16):
                        A = A*1e-2
                        B = B*1e-2
                    
                    C_ref = torch.matmul(A, B)
                    ith, jth = torch.randint(0, 128, (1,)).item(), torch.randint(0, 256, (1,)).item()
                    
                    # 尝试注入错误
                    # C_ref[ith, jth], success = utils.flip_infuse(C_ref[ith, jth], bit)
                    
                    # success = not success # 1-0 flip
                    
                    success = True # 用于测试误检率
                    
                    if not success:
                        continue
                    
                    valid_count += 1
                    
                    # 计算检查值和误差界限
                    c_check = torch.matmul(A, torch.sum(B, dim=-1, keepdim=True)).squeeze()
                    error_bound = utils.my_bound_improve_robust(A, B, dtype=dtype)
                    c_sum = torch.sum(C_ref, dim=-1, keepdim=True)
                    diff = torch.abs(c_check - c_sum.squeeze())
                    # print("diff_abs_mean:", diff.abs().mean().item())
                    
                    # 检查误差是否在界限内
                    result = (diff <= error_bound).all() and (diff != torch.inf).all() and ~torch.isnan(diff).any()
                    if not result:
                        error_count += 1
                        
                    # if result:
                        # print("c_sum:", c_sum[ith].item(), "c_check:", c_check[ith].item(), "diff:", diff[ith].item(), "bound:", error_bound[ith].item())
                
                # 检查该bit位是否有效
                if valid_count == 0:
                    print(f"Bit Position: {bit}, No successful injections after {max_attempts} attempts - SKIPPED")
                    f.write(f"Bit Position: {bit}, No successful injections after {max_attempts} attempts - SKIPPED\n")
                    continue
                
                # 计算错误率
                error_rate = error_count / valid_count * 100 if valid_count > 0 else 100
                print(f"Bit Position: {bit}, Valid Trials: {valid_count}, Errors Detected: {error_count}, Error Rate: {error_rate:.4f}%")
                f.write(f"Bit Position: {bit}, Valid Trials: {valid_count}, Errors Detected: {error_count}, Error Rate: {error_rate:.4f}%\n")
                
                total_rate += error_rate
                valid_bit_count += 1
            
            # 计算平均错误率
            if valid_bit_count > 0:
                avg_error_rate = total_rate / valid_bit_count
                print(f"Average Error Rate over all bits: {avg_error_rate:.6f}%\n")
                f.write(f"Average Error Rate over all bits: {avg_error_rate:.6f}%\n\n")
            else:
                print(f"No valid bits for this data type - all injections failed\n")
                f.write(f"No valid bits for this data type - all injections failed\n\n")
        
        f.write("\n" + "=" * 50 + "\n\n")

print("Results have been saved to 'error_injection_results.txt'")
