#!/bin/zsh

yes | brew update

brew install pyenv

brew install pyenv-virtualenv

brew install readline xz

alias brew='env PATH="${PATH//$(pyenv root)\/shims:/}" brew'

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
	
exec "$SHELL"

cat .python-version | pyenv install

cat .python-version | pyenv virtualenv $1 music_curator

pyenv activate music_curator

python3.10 -m pip install --upgrade pip

python3.10 -m pip install pip-tools

pip-compile --output-file=./requirements.txt ./requirements.in

pip3.10 install -r ./requirements.txt

# crontab -e
# 31 16 * * FRI /Users/jose/spotify-music-curator/start.sh