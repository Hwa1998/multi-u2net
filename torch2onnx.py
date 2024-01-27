import onnx
import torch.onnx
from model.u2net_onnx import U2NET, U2NETP

# out_ch 这个参数是 类别数量(background+分割类别)
u2net = U2NETP(3, 4)
print(u2net)
model_dir = "saved_models/u2netp/u2netp.pth"

if torch.cuda.is_available():
    u2net.load_state_dict(torch.load(model_dir))
    u2net.cuda()
    u2net.eval()
    dummy_input = torch.randn(1, 3, 600, 600).cuda()  # N C H W
else:
    u2net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    u2net.eval()
    dummy_input = torch.randn(1, 3, 600, 600)  # N C H W

# 关键参数 verbose=True 会使导出过程中打印出该网络的可读表示
torch.onnx.export(u2net, dummy_input, 'saved_models/u2netp/u2netp.onnx', verbose=True, opset_version=14)
onnx_model = onnx.load('saved_models/u2netp/u2netp.onnx')
onnx.checker.check_model(onnx_model)  # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))
