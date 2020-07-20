# disaster-response

### Table of contents

1. [Project Overview](#overview)
2. [Installation](#install)
3. [Data](#data)
4. [File Description](#file)
5. [Instructions](#instruct)
6. [Example Images](#images)
7. [License](#license)

## Project Overview <a name="overview"></a>

This project is a part of Udacity Data Scientist Nanodegree program. In this project, we apply different skills to analyze disaster data to build a API model to classify the disaster messager. The data is provided by Figure Eight.

Using the data, we build a machine learning pipeline to categorize the messages based on events and needs and then send it to concerned departments during disaster emergency.

In this project, we develop a web app where messages are categorized. The web app also display visualizations of data. 

## Installation <a name="install"></a>

The project code is written in HTML and Python. Jupyter Notebook is an option for better writing skills. The project requires python package like pandas, numpy, re, pickle, nltk, flask, ploty, sklearn, sqlalchemy, sys.

## Data <a name="data"></a>

The data set contains real messages that were sent during disaster events. The data is provided by [Figure Eight](https://appen.com/). The data is stored as two csv files namely `disaster_categories.csv` and `disaster_messages.csv`. The data files are stored in the data folder.

## File Description <a name="file"></a>

* **ETL Pipeline Preparation.ipynb**:  process_data.py development procces
* **ML Pipeline Preparation.ipynb**: train_classifier.py. development procces
* **process_data.py** - This python file reads the csv files from data folder and cleans the data. Later it stores the cleaned data to new database `DisasterResponse.db`. This file is stored in `data` folder.
* **train_classifier.py** - This python file contains the code for Machine Learning pipeline and builds models with the SQL data base. This file is stored in `models` folder.
* **run.py** - This python file contains code to initiate web app and also contains code for ploting charts.
* **templates folder** - contains html files.
* **example** - contains pdf of web app layout and examples of graphs and catergorized messages.

## Instructions <a name="instruct"></a>

1. After completion of all the code. Enter and run the following commands in the terminal to set up database and model.
      
      - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command to run web app.
  `python run.py`
  
3. Open another Terminal Window and run
  `env|grep WORK`

4. Go to `http://0.0.0.0:3001/`
   Or Go to `http://localhost:3001/`

5. If the 4th step doesn't work, you can browse on 
  `https://SPACEID-3001.SPACEDOMAIN`
  `SPACEID` is available when you run `env|grep WORK`.
  
## Example Images <a name="image"></a>

The example images are stored as pdf in `example` folder.

## License <a name="license"></a>

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model
