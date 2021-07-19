# NLP MBTI Personality Classification using an approximation algorithm for features selection.

## Dependencies:

pip install transformers
pip install tokenizers
pip install fsspec
pip install numpy
pip install scikit-learn
pip install torch, torchvision
pip install pandas

To run the code:<\b>
sudo python main.py -m **(model name)** -n 16(number of classes) -b **(batch size)** -l **(learning rate)** -e **(starting epoch)** -f **(number of epochs to run)** --optimizer SGD  --loss CCE --library hugging-face -s checkpoints --save_interval 10 -d ./mbti_1.csv --train --multi
