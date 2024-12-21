import torch


double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.short)
print(short_points.dtype)
print(double_points.dtype)

print("====== converting ========")
double_points = torch.zeros(10, 2, dtype=torch.short).double()
short_points = torch.ones(5, 4, dtype=torch.double).short()
print(double_points.dtype)
print(short_points.dtype)

print("===== shortcut converting =====")
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(4, 2).to(dtype=torch.short)
# to method will check if the convering is necessary, if yes, then it will convert else no.
print(double_points.dtype)
print(short_points.dtype)

print("====== auto converting ======")
double_points = torch.ones(5, 2, dtype=torch.double)
short_points = torch.ones(1, 5, dtype=torch.short)
short_points = short_points.transpose(0, 1)  # 变为 (5, 1)
print((double_points * short_points).dtype)
