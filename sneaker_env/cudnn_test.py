import torch
print(torch.cuda.is_available())  # Should say True
print(torch.backends.cudnn.enabled)  # Should say True
print(torch.backends.cudnn.version())  #