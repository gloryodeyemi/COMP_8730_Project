# Summarization of English-Yoruba Tweets with Code-Switches
The goal of this experiment is to summarize tweets with English and Yoruba code switches.

**Keywords:** Code-Switching, Tweet Summarization, Language Identification, Code-switching Detection, Translation, Natural Language Processing.

## The Data
The [twitter_data](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/Data/twitter_data.csv) csv file has three columns:
* **Tweets:** Tweets with code-switches.
* **Eng_source:** English source of the tweets. 
* **Summary:** Human annotated summary of the tweets.

## Requirements
You can find the modules and libraries used in this project in the [requirement.txt](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/requirements.txt) file. You can also run the code below.
```
pip install -r requirements.txt
```

## Structure
* **[Data](https://github.com/gloryodeyemi/COMP_8730_Project/tree/main/Data):** contains the data file used for this project.

* **[utils](https://github.com/gloryodeyemi/COMP_8730_Project/tree/main/utils):** contains the essential functions used for the project.

* **[data_analysis.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/data_analysis.ipynb):** A python notebook that uses the function in the utils to analyse the data used in this project. The results gives information about the data.

* **[data_collection.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/data_collection.ipynb):** A python notebook that shows you the procedure of collecting tweets from Twitter using the Twitter API and tweepy python library.

* **[quick_start.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/quick_start.ipynb):** A python notebook that shows a successful run of the project using the quickstart guideline.

* **[main.ipynb](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/main.ipynb) and [main.py](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/main.py)** are python notebook and script that utilizes the functions in utils to show the procedure of summarizing tweets with English-Yoruba code switches and the result gotten.

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
python main.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/gloryodeyemi/COMP_8730_Project/blob/main/LICENSE) file for details.

## Contact
Glory Odeyemi is currently undergoing her Master's program in Computer Science, Artificial Intelligence specialization at the [University of Windsor](https://www.uwindsor.ca/), Windsor, ON, Canada. You can connect with her on [LinkedIn](https://www.linkedin.com/in/glory-odeyemi-a3a680169/).

## References
1. [Tweepy](https://www.tweepy.org/)
