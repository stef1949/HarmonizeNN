
import torch, sys
print(sys.version)
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_arch_list())