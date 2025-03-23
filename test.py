import torch
a= torch.ones((64,1,32,64))
audio_last_layer = torch.nn.AdaptiveAvgPool2d((32,32))
b = audio_last_layer(a)
print(b.size())
