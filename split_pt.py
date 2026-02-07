import torch
import os

model_path = 'outputs/best.pt'
split_model_path = 'outputs/split/'
os.makedirs(split_model_path, exist_ok=True)

model = torch.load(model_path)

torch.save(model['model'], os.path.join(split_model_path, 'model.pt'))
torch.save(model['head'], os.path.join(split_model_path, 'head.pt'))