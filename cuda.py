import torch
print(torch.cuda.is_available())  # Should now be True
print(torch.cuda.get_device_name(0))  # Should show your GPU model