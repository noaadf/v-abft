import utils    
import torch
import torch_npu
import math

n = 4096
trials = 30000
dtype = torch.bfloat16
rate = 0
flag = 0
for bit in range(7, 15):
    error_count = 0
    valid_count = 0
    for _ in range(trials):
        A = utils.generate_matrice_clamp(128, n, std=1, mean=0, device=utils.device_npu, dtype=dtype)
        B = utils.generate_matrice_clamp(n, 256, std=1, mean=0, device=utils.device_npu, dtype=dtype)
        C_ref = torch.matmul(A, B)
        ith, jth = torch.randint(0, 128, (1,)).item(), torch.randint(0, 64, (1,)).item()
        C_ref[ith, jth], success = utils.flip_infuse(C_ref[ith, jth], bit) #测试误检率时注释这一行，使用下一行
        # success = True
        if not success:
            continue
        valid_count += 1
        c_check = torch.matmul(A, torch.sum(B, dim=-1, keepdim=True)).squeeze()
        error_bound = utils.my_bound_improve_robust(A, B)
        
        c_sum = torch.sum(C_ref, dim=-1, keepdim=True)
        diff = torch.abs(c_check - c_sum.squeeze())
        
        result = (diff <= error_bound).all() and (diff != torch.inf).all() and ~torch.isnan(diff).any()
        if not result:
            error_count += 1
    print(f"Bit Position: {bit}, Valid Trials: {valid_count}, Errors Detected: {error_count}, Error Rate: {error_count/valid_count*100 if valid_count > 0 else 0:.4f}%")
    rate += error_count/valid_count*100 if valid_count > 0 else 100
print(f"Average Error Rate over all bits: {rate/8:.6f}%")