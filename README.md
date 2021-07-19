# NLP MBTI Personality Classification using an approximation algorithm for features selection.

## Dependencies:

pip install transformers <br>
pip install tokenizers <br>
pip install fsspec <br>
pip install numpy <br>
pip install scikit-learn <br>
pip install torch, torchvision <br>
pip install pandas <br>

## To run the project:<br>
sudo python main.py -m **(model name)** -n 16(number of classes) -b **(batch size)** -l **(learning rate)** -e **(starting epoch)** -f **(number of epochs to run)** --optimizer **(choose an optimizer among SGD, ADAM, ADAMW)**  --loss **(choose a loss criteria among CCE, MML, BCE, MSE)** --library **(Library to pick pre trained models)** -s **(directory to save the results)** --save_interval **(number of epochs to save the results)** -d **(path of the dataset)** --train **(to set the model in train mode)** --multi **(to use multiple GPUs for computation)**
<br>

## Libraries used:<br>

**-** Numpy for mathematical computations <br>
**-** Argparse for arguments parsing<br>
**-** OS to disable tokenizers parallelization<br>
**-** Pandas for data handling<br>
**-** Torch for executing neural network functions<br>
**-** Kmeans_pytorch to implement k-means with cuda enabled<br>
**-** Transformers to use pre trained models<br>
**-** Fairscale to use Sharded gradient scaler<br>
**-** Uuid to assign unique identifier to the model<br>
