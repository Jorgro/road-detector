PYTHONPATH=/cluster/home/jorgro/road-detector
python ./tools/test.py ./configs/road_detector_yolox.py ./runs/latest.pth --work-dir ./results --out ./results/yolox.pkl
python ./tools/analysis_tools/analyze_results.py ./configs/road_detector_trident.py ./results/yolox.pkl ./result
python ./tools/analysis_tools/eval_metric.py ./configs/road_detector_trident.py ./results/yolox.pkl --eval mAP
