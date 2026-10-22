import torch
import torchvision
from torchvision.models import swin_t

model = swin_t(weights=None)
model.features[0][0] = torch.nn.Conv2d(13, 96, kernel_size=(4, 4), stride=(4, 4))
x = torch.randn(2, 13, 32, 32)
feat = model.features(x)
print("Features shape:", feat.shape)

out = model(x)
print("Output shape:", out.shape)
