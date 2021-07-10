# Some standard imports
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from rosbot_model import RosbotModel

model = RosbotModel(        
        n_inputs=5,                     
        n_outputs=4,
        n_layers=1,
        hidden_size=64,
        activation_function='relu',
        learning_rate=0.005,
        linear=True
    )
state_dict = torch.load('model_1q1lyfha.pt')
model.load_state_dict(state_dict)
model.eval()
batch_size = 50
# Input to the model
x = torch.randn(batch_size, 5, requires_grad=True)
print(x)
torch_out = model(x)

# Export the model
torch.onnx.export(
    model,                     # model being run
    x,                         # model input (or a tuple for multiple inputs)
    "model.onnx",              # where to save the model (can be a file or file-like object)
    export_params=True,        # store the trained parameter weights inside the model file
    opset_version=10,          # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names = ['input'],   # the model's input names
    output_names = ['output'],  # the model's output names
    dynamic_axes={
        'input' : {0 : 'batch_size'},
        'output' : {0 : 'batch_size'}
    }
)

# test
import nnio

model_onnx = nnio.ONNXModel("model.onnx")
onnx_out = model_onnx(x[5:10].detach().numpy())
print(x.detach().numpy().dtype)

print("torch out {}".format(torch_out))
print("onnx out {}".format(onnx_out))