# NLP MBTI Personality Classification using an approximation algorithm for features selection.

## Dependencies:

pip install transformers
pip install tokenizers
pip install fsspec
pip install numpy
pip install scikit-learn
pip install torch, torchvision
pip install pandas

To run the code:<br>
sudo python main.py -m **(model name)** -n 16(number of classes) -b **(batch size)** -l **(learning rate)** -e **(starting epoch)** -f **(number of epochs to run)** --optimizer **(choose an optimizer among SGD, ADAM, ADAMW)**  --loss **(choose a loss criteria among CCE, MML, BCE, MSE)** --library **(Library to pick pre trained models)** -s **(directory to save the results)** --save_interval **(number of epochs to save the results)** -d **(path of the dataset)** --train **(to set the model in train mode)** --multi **(to use multiple GPUs for computation)**
