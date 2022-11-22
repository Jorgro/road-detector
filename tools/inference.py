from mmdet.apis import init_detector, inference_detector
import mmcv
import glob

# Specify the path to model config and checkpoint file
config_file = 'configs/road_detector_yolo.py'
checkpoint_file = 'runs/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
test_dataset = glob.glob("/cluster/home/jorgro/datasets/Norway/test/images/*.jpg")
save_file = "/cluster/home/jorgro/submission.txt"


#with open(save_file, 'w') as f:
for img in test_dataset:
    result = inference_detector(model, img)
    string = f"{img}"
    for i, obj in enumerate(result):
        if obj.shape[0]:
            string += f" {i} {obj[0]} {obj[1]} {obj[2]} {obj[3]}"
    print(string)
       # f.write(f"{}")
