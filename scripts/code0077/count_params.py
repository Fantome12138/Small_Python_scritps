'''在yolov7的detect.py最后加上即可'''

import thop
device = select_device(opt.device)
model = attempt_load(opt.weights,map_location = device)
input1 = torch.rand(1,3,640,640).to(device)
total_ops, total_params = thop.profile(model,(input1,))
Params = total_params/(1000**2)  # M
FLOPs = (total_ops/(1000**3))*2  # G
print(f'Params:{round(Params,2)}M\nFLOPs:{round(FLOPs,2)}G')