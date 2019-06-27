import os
import pathlib

os.system(f'export PYTHONPATH={pathlib.Path(os.getcwd()).parent}')

if __name__ == '__main__':
    cmd = 'nohup python train.py' \
          ' --input=../datasets/result' \
          ' --batch_size=256 --gpu_index=2' \
          ' --dropout=0.2 --vip=10.61.1.245' \
          ' --mode_type=b --hidden_size=128' \
          ' --model_name=v9.0_dp2_128' \
          ' >v9.0_dp2_128.txt 2>&1 &'

