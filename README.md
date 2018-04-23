# MNIST Classification with Deep Convolutional Neural Networks
A deep convolutional neural network implemented with TensorFlow to classify the MNIST dataset.

## Code
Code for this project is contained in the ```src/``` directory. There are two different models, one in 
```src/mnist-single``` and one in ```src/mnist-deepcnn```. 

The model in ```src/mnist-deepcnn/mnist-deepcnn.py``` is the main model. After descending to the 
```src/mnist-deepcnn/``` directory, it can be run using the command ```python3 mnist-deepcnn.py```. This trains the 
deep neural network and evaluates it, printing out the accuracy.

If you wish to run the pretrained model, once again descend into the ```src/mnist-deepcnn/``` directory and run it 
with the command ```python3 mnist-deepcnn-pretrained.py```. The pretrained model extracts data from 
```src/mnist-deepcnn/savefiles``` and evaluates the accuracy of the model.

The ```mnist-single.py``` and ```mnist-single-pretrained.py``` files can be run in a similar manner from their 
directory: ```src/mnist-single/```.

## Data
At the beginning of each of the python scripts, there is a line that downloads the data into the root level directory 
of the repository (if run from the same directory as the script). The rest of the script can find the data and use it
automatically.

## Paper
The paper can be found in the ```paper/``` directory. The full filename is ```paper/nips_2017.pdf```.

The ```.tex``` was compiled on guinness from the ```paper/``` directory using the sequence of commands 
```pdflatex nips_2017.tex```, ```bibtex nips_2017.tex```, ```pdflatex nips_2017.tex```, ```pdflatex nips_2017.tex```
after which ```nips_2017.pdf``` had the complete output 
