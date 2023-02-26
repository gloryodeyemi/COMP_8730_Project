# Summarization of English-Yoruba Tweets with Code-Switches
The goal of this experiment is to summarize tweets with English and Yoruba code switches.

**Keywords:** Code-Switching, Tweet Summarization, Language Identification, Code-switching Detection, Translation, Natural Language Processing.

## The Data
The [twitter_data](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/Data/twitter_data.csv) csv file has three columns:
* **Tweets:** Tweets with code-switches collected from Twitter using the Twitter API with Twitter developer account credentials and a python library - [Tweepy](https://www.tweepy.org/). 
* **Eng_source:** Human annotated English translation of the tweets. 
* **Summary:** Human annotated summary of the tweets.

## Requirements
You can find the modules and libraries used in this project in the [requirement.txt](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/requirements.txt) file. You can also run the code below.
```
pip install -r requirements.txt
```

## Structure
* **[Data](https://github.com/gloryodeyemi/COMP_8730_Project/tree/main/Data):** contains the data file used for this project.

* **[utils](https://github.com/gloryodeyemi/COMP_8730_Project/tree/main/utils):** contains the essential functions used for the data analysis.

* **[data_analysis.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/data_analysis.ipynb):** A python notebook that uses the function in the utils to analyse the data used in this project. The results gives information about the data.

* **[data_collection.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/data_collection.ipynb):** A python notebook that shows you the procedure of collecting tweets from Twitter using the Twitter API and tweepy python library.

* **[quick_start.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/quick_start.ipynb):** A python notebook that shows a successful run of the project using the quickstart guideline.

* **[Summarization.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/Summarization.ipynb) and [Summarization.py](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/Summarization.py)** are python notebook and script that shows the procedure of summarizing tweets with English-Yoruba code switches and the result gotten.

## Quickstart Guideline
1. Clone the repository
``` 
git clone https://github.com/gloryodeyemi/COMP_8730_Project.git 
```
2. Change the directory to the cloned repository folder
```
%cd .../COMP_8730_Project
```
3. Install the needed packages
```
pip install -r requirements.txt
```
4. Run the script
```
python Summarization.py
```

## Contact
Glory Odeyemi is currently undergoing her Master's program in Computer Science, Artificial Intelligence specialization at the [University of Windsor](https://www.uwindsor.ca/), Windsor, ON, Canada. You can connect with her on [LinkedIn](https://www.linkedin.com/in/glory-odeyemi-a3a680169/).

## References
1. [Tweepy](https://www.tweepy.org/)
