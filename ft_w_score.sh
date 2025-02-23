#!/bin/bash

# python3 train.py --cfg-path ./configs/ft_assess_with_score_config.yaml

python3 train.py --cfg-path ./configs/ft_assess_with_score_config_test.yaml

python3 inference_jsons.py --cfg-path ./configs/ft_assess_with_score_config_test.yaml --dataset-cfg ./configs/dev_set.yaml