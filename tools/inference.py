from mmdet.apis import init_detector, inference_detector
import mmcv
import glob

# Specify the path to model config and checkpoint file
config_file = 'configs/road_detector_yolo.py'
checkpoint_file = 'runs/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
test_dataset = sorted(glob.glob("/cluster/home/jorgro/datasets/Norway/test/images/*.jpg"))
save_file = "/cluster/home/jorgro/submission.txt"

with open(save_file, 'w') as f:
    for j in range(len(test_dataset), 8):
        results = inference_detector(model, test_dataset[j:j+8])
        for ind, result in enumerate(results):
            string = f"{test_dataset[j+ind][49:]}"
            for i, obj in enumerate(result):
                if obj.shape[0]:
                    string += f" {i+1} {int(obj[0][0])} {int(obj[0][1])} {int(obj[0][2])} {int(obj[0][3])}"
            string += "\n"
            f.write(string)
            print(string)
       # f.write(f"{}")
