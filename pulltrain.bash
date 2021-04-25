#!/bin/bash
git pull
python train.py --config=Colab --misc.save_comment=$1
