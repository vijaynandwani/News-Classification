# Classification of News HeadLines

##### [DEMO VIDEO](https://www.youtube.com/watch?v=HeKchZ1dauM&feature=youtu.be)

News Headline Classification through multiple machine learning model and comparison of results.

Models implemented:

 * Multinomial Naive Bayes 
 * Support Vector Machines 
 * Neural Network with Softmax Layer
 

Metrics used to evaluate the performance of models:

 * Precision
 * Recall
 * F1 Score 
 
 We evaluate each classifier's ability to select the appropriate category given an article’s title and a brief article description. The confusion matrix is created to  explore the results and calculate the metrics. 



###### Feature Extraction Techniques:
The collection of text documents is converted to a matrix of token counts using count vectorize that produces a sparse representation of the counts.

TFIDF,term frequency–inverse document frequency, is the statistic that is intended to reflect how important a word is to a document in our corpus. This is used to extract the most meaningful words in the Corpus. 

###### Link to Dataset: [News Article Dataset](http://acube.di.unipi.it/tmn-dataset/) 
TagMyNews Datasets is a collection of datasets of short text fragments that we used for the evaluation of  our topic-based text classifier. This is a dataset of  ~32K english news extracted from RSS feeds of popular newspaper websites (nyt.com, usatoday.com, reuters.com). Categories are: Sport, Business, U.S., Health, Sci&Tech, World and Entertainment.



Packages required: 

 * Pandas
 * sklearn
 * Numpy
 
 
 
![Multinomial Naive Bayes](https://i.imgur.com/2gaK9iO.png)
![Softmax](https://i.imgur.com/R2XHiuB.png)
![SVM](https://i.imgur.com/dvfwxY8.png)
![Average of three](https://i.imgur.com/1WtrPRv.png)






