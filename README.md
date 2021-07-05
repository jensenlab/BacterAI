# BacterAI

Activate virtual env:
source /home/lab/.local/share/virtualenvs/BacterAI-b4eJG3kt/bin/activate

Monitor GPU usage: 
watch -d -n 0.1 nvidia-smi

pipenv path:
/home/lab/.local/bin/pipenv

Command for profiling:
sudo LD_LIBRARY_PATH=/usr/local/cuda/extra/CUPTI/lib64:$LD_LIBRARY_PATH /home/lab/.local/share/virtualenvs/BacterAI-b4eJG3kt/bin/python neural.py

Start tensorboard:
tensorboard --port 6006 --bind_all --logdir=tensorboard_logs/

View tensorboard through ssh:
ssh -L 6006:127.0.0.1:6006 lab@ghost.bioe.illinois.edu

ssh -L 8000:127.0.0.1:8000 lab@ghost.bioe.illinois.edu


Install gurobipy:
cd /opt/gurobi901/linux64
which python3 
sudo /home/lab/.local/share/virtualenvs/BacterAI-b4eJG3kt/bin/python3 setup.py install #Change python3 path
cd ~/Documents/github/BacterAI