# Project Link:  [Billboard-Hot-100-Hits-Prediction](https://billboard-hits-prediction.herokuapp.com)

# Billboard Hot 100 Hit Prediction
:notes: Predicting Billboard's Year-End Hot 100 Songs using audio features from Spotify data.

## Overview
Each year, Billboard publishes its Year-End Hot 100 songs list, which denotes the top 100 songs of that year. The objective of this project was to see whether or not a machine learning classifier could predict whether a song would become a hit given its intrinsic audio features as well as lyrics.

The goal of this project is to see if a song's audio characteristics and lyrics can determine a song's popularity. Data and analytics aside, music listeners around the world probably have seen music trends change over time. Although each listener has custom interests in music, it is pretty clear when we listen to a hit song or soon to be hit song (consider Old Town Road). And over time, we see the characteristics of hit songs change. So, rather than using our intuition or "gut-feeling" to predict hit songs, the purpose of the project is to see if we can use intrinsic music data to identify hits.

Hit Song Science can help music producers and artists know their audience better and produce songs that their fans would love to hear. Artists can better know what lyrics to write and tune the meaning of their song to what their fanbase would enjoy. Additionally, audio engineers can work with musicians to tweak intrinsic music qualities to make a song more popular catchy and likable.

Also, it can highlight unknown artists whose music is characteristic of top songs on the Billboard Hot 100. This allows underground artists (i.e. Lil Tecca), who might not have the publicity help from an agency or a record label, to have a chance at gaining recognition. 


## Data and Features
A sample of songs was downloaded from [Kaggle](https://www.kaggle.com/danield2255/data-on-songs-from-billboard-19992019/download) and [The million songs](http://millionsongdataset.com/pages/getting-dataset/) which included songs from various albums. Concatenated these two datasets into a bigger one which consists of 9227 songs.
The dataset consists of these audio features which was used to predict the hit of a song.
 - **Mood**: Danceability, Valence, Energy, Tempo
- **Properties**: Loudness, Speechiness, Instrumentalness
- **Context**: Liveness, Acousticness


After cleaning the data, a dataset of approx. 8653 songs was created.

![](images/data-distribution.png)

In the above graph, you can see that maximum songs fail to get into Top100 list. Only 16.4% songs appear in Top100 list.

![](images/fig4.png)

![](images/fig5.png)

![](images/fig6.png)

![](images/freq-vs-decade.png)

From the graph, it is observed maximum songs are released between 2010-2020.

![](images/genre-dist.png)

![](images/genre-dist1.png)

Most of the songs released are of 'Pop', 'Rap' and 'Rock' genre. Half of the songs released are of Genre 'Pop'.

## Exploratory Data Analysis

**Distribution of Genre

![](images/distgen.png)

The distribution of release year is negatively skewed

**Spotify Features over Time for each decade**

![](images/fig1.png)

![](images/fig2.png)

![](images/fig3.png)

CONCLUSIONS FROM THE PLOTS :

1.For a song to hit list of billboard 100 top songs, its danceability must be above than 0.6.

2.Songs with low loudness level, have more chances to hit billboard top 100 list.

3.Songs with acousticness level between 0.05 to 0.2 ,generally hits the billboard top 100 list.

**Correlation between each feature**

![](images/corr.png)

There is good positive correlation between Loudness and Energy. While Acoustiness and Energy are negatively correlated. Also there is negative correlation between Loudness and Acoustiness.

**Feature Comparisons**

![](images/comp1.png)

![](images/comp2.png)

Scatter plot shows that loud songs with High Energy makes up on the billboard Top100.

![](images/comp3.png)

Rap songs preffered for dancing. While Metal types are least preffered.

**Feature distributions**

![](images/distoffeat.png)

It is showing the distribution of each feature

**Audio Features vs Top100(Density)**

![](images/feattar.png)

It seems that Valance, Tempo, key are not much significant features to predict if the song will be on Billboard or not. While Instrumentalness, Loudness and Energy seems to be significant.



## Models
Given the unbalanced nature of the dataset, used SMOTE to balance it. Used Standardization technique to scale it down into one scale. So, in addition to aiming for high accuracy, another objective of modeling is to ensure a high AUC (so that TPR is maximized and FPR is minimized). The AUC tells us how well the model is capable of distinguishing between the two classes.

Here's a list of all the models I tested:
  1. Decision Tree 
  2. Improved Decision Tree (with hyperparameter tuning)
  4. Improved LDA (with hyperparameter tuning)
  5. Decision Tree with Adaboost (with hyperparameter tuning)
  
**Model Summaries:**

| Model   | Accuracy   | AUC   |
| -----   | :--------: | :---: |
| Decision Tree | 0.74 | 0.77 |
| Improved Decision Tree | 0.75 | 0.79 |
| Improved LDA | 0.65 | 0.70 |
| Decision Tree with Adaboost | 0.83 | 0.90 |




**Model Summary:**

| Accuracy   | AUC   |
| :--------: | :---: |
| 0.83 | 0.90 |

The Decision tree-Adaboost with hyper parameter tuning model gave the highest accuracy score and AUC score.

#### Deployment:
With the help of [Flask](https://flask.palletsprojects.com/en/2.0.x/) framework, I have created a webpage (link given on the top) which takes audio features as input and predict whether the song wiil hit Billboard or not? Also used HTML and CSS to design the webpage. Below are some screenshots od webpage:

![](images/deploy2.png)

![](images/deploy3.png)

you have to press the Predict button after everything is done.

![](images/deploy1.png)

And the Prediction page will look like this.



## Conclusion
The best model after testing seems to (improved) logistic regression and bagging. Both these models yielded high accuracy (~81%) and they had an above average TPR (~0.3) and AUC (~0.785). Also, the stacked model did a good job of minimizing FPR and helped increase the AUC (~0.80).

## Future Work
- Append more music awards (Grammy, Apple Music Awards, iHeartRadio Music Awards, etc.) to balance dataset of "hit" songs
- Reduce time window (2-3 years) or prepare a time-series model
- Build deep learning model

## Sources
1. https://www.kaggle.com/edalrami/19000-spotify-songs
2. https://developer.spotify.com/discover/
3. https://developer.musixmatch.com
4. https://en.wikipedia.org/wiki/Hit_Song_Science
5. https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
6. https://stats.stackexchange.com/questions/179864/why-does-shrinkage-work
7. https://statweb.stanford.edu/~jtaylo/courses/stats203/notes/penalized.pdf
8. https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9
9. https://towardsdatascience.com/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f
