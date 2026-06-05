#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' '8s/.*/STUDY_NAME = "Type2"/' config.py
    sed -i '' '13s/.*/METHOD = "genetic"/' config.py
else
    sed -i '8s/.*/STUDY_NAME = "Type2"/' config.py
    sed -i '13s/.*/METHOD = "genetic"/' config.py
fi

python -u main.py --type genetic