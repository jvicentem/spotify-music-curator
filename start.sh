#!/bin/zsh

cd /Users/jose/spotify-music-curator

pyenv activate music_curator

python ./spoti_curator/app.py

if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

echo $SPECIAL_ENV | sudo -S -k pmset repeat wakeorpoweron F 16:30:00

echo $SPECIAL_ENV | sudo -S -k pmset sleepnow
