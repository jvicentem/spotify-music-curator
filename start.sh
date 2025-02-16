#!/bin/zsh

cd ~/spotify-music-curator

cd ./spoti_curator
eval $(cat .env | sed 's/^/export /')
cd ..

source $CONDA_ENV_PATH

conda activate spoti_curator

export PYTHONPATH=$PYTHONPATH:$PWD

python ./spoti_curator/app.py

echo $SPECIAL_ENV | sudo -S -k pmset repeat wakeorpoweron S 12:30:00 # for the world project

echo $SPECIAL_ENV | sudo -S -k pmset sleepnow
