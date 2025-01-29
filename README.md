# Cloning the repository

Suggested command to clone the repository:   
“git clone \--branch main-stage-3 \--single-branch [https://github.com/andrucior/ML.git](https://github.com/andrucior/ML.git)”  
Estimated download size: 350MB

# Data processing

To process the audio files class *DataProcessor* (from file *DataProcessor.py*) is used. ***DataProccessor*** **only works with audio, and does not create spectrograms*.***

## Processing all dataset

To process the whole dataset we use the function *DataProcessor.proccess\_data(extract\_path, proccessed\_data).* Function will process all audio files in the directory, and process them.

## Processing function

Each audio file must go through several processing functions. List of the functions:

- *DataProccessor.check\_for\_low\_variance(audio, variance\_trashold)*: this function will check if the audio has low variance.  
- *DataProccessor.remove\_silence(audio, top\_db)*: this function will remove silence from the audio.  
- *DataProccessor.append\_to\_one\_seccond(audio, sr):* this function will append audio to one second length script.

# Spectrogram creation

Spectrograms created using class *SpectrogramCreator* (from file *SpectrogramCreator.py*).  
*SpectrogramCreator.create\_spectrogram(file\_path)*: will create a spectrogram for a given file.

# Creating spectrograms and processing whole dataset

If you run file *SpectrogramGeneratorScript.py* it will process all audio files and create spectrograms for them. You should only define 3 variables representing directories at the beginning of the file.

# Creating small spectrograms

If you run file *smallSpectrogramCreatorScript.py* it will transform all spectrograms in the given directory into smaller spectrograms and save the in provided directory. You should only define 2 variables representing directories at the beginning of the file.

# Model parameter descriptions and test results 

Each subfolder in the folder **models** contains the following files and sub catalogues:

* /analysis \- folder containing training analysis  
* /test\_analysis \- folder containing results of test analysis including confusion matrix and other charts described in the project rapport  
* description.txt \- file including model training parameters and short notes and comments  
* {model\_name}.pth \- file containing trained model

For models based on networks with MC dropout (model name includes word dropout) there are test results for each testing mode (as described in the report) as well as images with combined charts for those modes.

There is also an **ensemble** subfolder. It contains test results for various combinations of models tested, labeled 1-9 in chronological order of tests. Apart from test results each folder contains the models\_list.txt file which includes:

* Network classes of the models used  
* Names of the models used  
* Weight, each model had

# Model files

List of files containing models:

- *model\_net.py*: the simplest model, designed to work with not compressed spectrograms.  
- *small\_model\_net.py*: analog of *model\_net.py*, designed to work with compressed spectrograms.  
- *small\_model\_net\_extra\_layers.py*: model containing additional layers, 4 convolutional and 3 pooling layers. Designed to work with compressed spectrograms.

# Model training

To train model you should run file *small\_model\_train\_script.py* (for models starting with “*small\_*”) or *model\_train\_script.py* (for *model\_net.py*).

## *small\_model\_train\_script.py*

To properly run this file the following things should be defined:

- Variable *train\_directory*: path to training dataset (now located at line: 63\)  
- Variable *val\_directory*: path to validation dataset (now located at line: 64\)  
- Variable *net*: model you want to train, it should be imported and than written inside of function, and then provided to this variable (*ex: net \= SmallNet().to(device)*) (now located at line: 91\)

## *model\_train\_script.py*

To properly run this file the following things should be defined:

- Variable *train\_directory*: path to training dataset (now located at line: 41\)  
- Variable *net*: model you want to train, it should be imported and than written inside of function, and then provided to this variable (*ex: net \= Net().to(device)*) (now located at line: 48). **This variable have already been set to model from *model\_net.py*, and should not be changed, as *model\_train\_script.py* used to train only this model.**

# Model testing

To test model you should run file *small\_model\_test\_script.py* (for models starting with “*small\_*”) or *model\_test\_script.py* (for *model\_net.py*).

##  *model\_test\_script.py*

To properly run this file the following things should be defined:

- Variable *test\_directory*: path to testing dataset (now located at line: 17\)  
- Variables *model\_name, model\_path*: name of model (*.pth* file) you want to test, and path to a model directories (*model\_path* usually should not be changed) (now located at lines: 13, 16\)

##  *model\_test\_script.py*

To properly run this file the following things should be defined:

- Variable *test\_directory*: path to testing dataset (now located at line: 17\)  
- Variables *model\_name, model\_path*: name of model (*.pth* file) you want to test, and path to a model directories (*model\_path* usually should not be changed)

# Monte Carlo dropout

**Networks:**

* *MC\_small\_net.py* \- network with one dropout layer  
* *MC\_small\_net2.py* \- network with two dropout layers

*MC\_small\_model\_test\_script.py* \- testing script for each testing mode as described in the report.
