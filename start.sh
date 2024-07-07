#!/bin/zsh

cd ~/spotify-music-curator

cd ./spoti_curator
eval $(cat .env | sed 's/^/export /')
cd ..

export PYTHONPATH=$PYTHONPATH:$PWD

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate music_curator

python ./spoti_curator/app.py

echo $SPECIAL_ENV | sudo -S -k pmset repeat wakeorpoweron F 16:30:00

echo $SPECIAL_ENV | sudo -S -k pmset sleepnow
