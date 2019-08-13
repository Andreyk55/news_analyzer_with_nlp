- You should run news_analyzer.py.

- Project perpuse: get headlines from a news website and perform a sentiment analysis on them using nlp analysis with machine learning. 


This project does the following:

1. Gets an html page of an Hebrew news website. 
2. Extracts news titles from that page.
3. Translates the data into English.
4. Trains nlp classifiers.
5. Performes a sentiment analysis on each news headline.


Extensions which do not appear in this version and can be added without much efford:
1. Collect data from several websites.
   Notice - need to write additional functions in otder to collect&clean the data.
2. Extract additional data.
3. Modify the translation module in order to get unlimited translation.
   Currently, baecause of google translation limitation I have commented the data extraction from the website and now working with files.
   The task will be simpler when working with English data beacause nlp library analyzes English text.
   Another solution is to work with Hnltk (Hebrew nlk) but for my knowlege it is an academic library.
4+5. The nlp module is based on :
     YouTube videos- "NLTK with Python 3 for Natural Language Processing" by sentdex (reccomended).
    Please read details bellow about the nlp analysis.


NLP analysis:
- As mentioned above the code is based on a youtube course.
- The classifier which is used is an aggregation of several classifiers and most of them use machine learning algoritms.
- The training used classified movie reviews data.
- IMPORTANT NOTE - I have trained the classifiers with small amount of data baecause of limitations in my computer.
  To get better results follow the instruction specified in the sentiment_analysis.py module and train with grater data.
- Reasons for bias:
  - Small training set actually used.
  - Maybe there is a better data to train the classifiers with (currently used movie reviews).
  - The headlines are translated from Hebrew, maybe this can cause a bias.
  - Maybe there are additional reasons which i am not aware of baecause this is my first encounter with nlp.


Results from Tests:
- Poor baecause of great bias in nlp analysis, can fix this as explained above.

   
Conclusion:
- This project implements several intersesting building blocks which can be axpanded and used in other projects.
- The most important thing is that I enjoyed working on this project.
