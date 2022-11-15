PYTHONPATH=/cluster/home/jorgro/road-detector python ./tools/analysis_tools/analyze_results.py ./configs/road_detector_trident.py ./results/yolox.pkl ./result
PYTHONPATH=/cluster/home/jorgro/road-detector python ./tools/analysis_tools/eval_metric.py ./configs/road_detector_trident.py ./results/yolox.pkl --eval mAP
