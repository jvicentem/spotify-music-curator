#!/bin/zsh

cd /Users/jose/spotify-music-curator

pyenv activate music_curator

cd ./spoti_curator

eval $(cat .env | sed 's/^/export /')

cd ..

python ./spoti_curator/app.py

echo $SPECIAL_ENV | sudo -S -k pmset repeat wakeorpoweron F 16:30:00

echo $SPECIAL_ENV | sudo -S -k pmset sleepnow
