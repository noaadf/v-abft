import utils    
import torch
import torch_npu
import math

n = 1024
trials = 3000
dtype = torch.bfloat16
rate = 0
flag = 0
for bit in range(15, 16):
    error_count = 0
    valid_count = 0
    for _ in range(trials):
        A = utils.generate_matrice_clamp(128, n, mean=0, std=1, device=utils.device_npu, dtype=dtype)
        B = utils.generate_matrice_clamp(n, 256, mean=0, std=1, device=utils.device_npu, dtype=dtype)
        C_ref = torch.matmul(A, B)
        ith, jth = torch.randint(0, 128, (1,)).item(), torch.randint(0, 256, (1,)).item()
        C_ref[ith, jth], success = utils.flip_infuse(C_ref[ith, jth], bit)  #测试误检率时注释这一行，使用下一行
        # success = True
        if not success:
            continue
        valid_count += 1
        
        c_check, error_bound = utils.checksum_bound_high_precision(A, B, C_ref)
        c_sum = torch.sum(C_ref, dim=-1, keepdim=True, dtype=torch.float32)
        diff = torch.abs(c_check - c_sum.squeeze())
        # if flag == 0:
        #     print(f"diff: {diff}")
        #     print(f"error_bound: {error_bound}")
        #     flag = 1
        result = (diff <= error_bound).all() and (diff != torch.inf).all() and ~torch.isnan(diff).any()
        # if result:
        #     print("c_sum:", c_sum[ith].item(), "c_check:", c_check[ith].item(), "diff:", diff[ith].item(), "bound:", error_bound[ith].item())
        if not result:
            error_count += 1
    print(f"Bit Position: {bit}, Valid Trials: {valid_count}, Errors Detected: {error_count}, Error Rate: {error_count/valid_count*100 if valid_count > 0 else 0:.4f}%")
    rate += error_count/valid_count*100 if valid_count > 0 else 0
print(f"Average Error Rate over all bits: {rate/8:.4f}%")
    
