# Project Link:  [Billboard-Hits-Prediction](https://billboard-hits-prediction.herokuapp.com)

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



## Models and Results
Given the unbalanced nature of the dataset, any model chosen would automatically yield high accuracy. So, in addition to aiming for high accuracy, another objective of modeling is to ensure a high AUC (so that TPR is maximized and FPR is minimized). The AUC tells us how well the model is capable of distinguishing between the two classes.

Also, after EDA, I decided to only consider songs released between 2000-2018 because it is evident that music trends and acoustic features change over time, and song characteristics of the '90s would probably be not reflective of '00s and '10s decades. *(Note: For the sake of sample size I decided to combine '00s and '10s decades together. However, with the conglomeration of more songs and awards, it is probably better to consider a smaller time window)*

Here's a list of all the models I tested:
  1. Logistic Regression
  2. Improved Logistic Regression (with un-important Spotify features removed)
  3. LDA
  4. 10-fold CV CART
  5. Random Forest
  6. Bagging
  7. 10-fold CV KNN
  
**Model Summaries:**

| Model   | Accuracy   | TPR   | AUC   |
| -----   | :--------: | :---: | :---: |
| Baseline | 0.798 | na | na |
| Logistic Regression | 0.809 | 0.289 | 0.786 |
| **Improved Logistic Regression** | **0.810** | **0.300** | **0.785** |
| LDA | 0.805 | 0.280 | 0.774 |
| 10-fold CV CART | 0.805 | 0.123 | 0.706 |
| Random Forest | 0.813 | 0.174 | 0.7731 |
| **Bagging** | **0.818** | **0.300** | **0.785** |
| 10-fold CV KNN | 0.801 | 0.014 | 0.736 |

### Additional Modeling

#### Stacking:
Additionally, I tested out an ensemble method by stacking a few models together (logistic + LDA + CART). Model ensembling is a technique in which different models are combined to improve predictive power and improve accuracy. Details regarding stacking and ensemble methods can be found [here](https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html).

**Model Correlation Matrix:**

|     | lda | rpart | glm |
| --- | --- | :-----: | :---: |
| **lda** | 1.0000000 | 0.1656148 | 0.9283705 |
| **rpart** | 0.1656148 | 1.0000000 | 0.2025172 |
| **glm** | 0.9283705 | 0.2025172 | 1.0000000 |

![](/images/stack-scatter.png)

**Model Summary:**

| Accuracy   | TPR   | AUC   |
| :--------: | :---: | :---: |
| 0.814 | 0.297 | 0.797 |

The stacked model achieved high accuracy and TPR that is comparable to the improved logistic regression and bagging model. However, more importantly, the stacked model greatly improved the AUC.

#### Penalized Regression:
Due to a large number of features (Spotify features + lyrics bag-of-words), I decided to use a penalized logistic regression model. This imposes a penalty to the logistic model for having too many variables. This results in lowering the dimensionality of the feature spacing by shrinking the coefficients of the less important features toward zeros. I specifically used the following penalized regression techniques:

- **Ridge Regression**: all the features are included in the model, but variables with minor contribution have their coefficients close to zero
- **Lasso Regression**: the coefficients of less contributive features are forced to zero and only the most significant features are kept

(An explanation regarding penalty methods and shrinkage can be found [here](https://stats.stackexchange.com/questions/179864/why-does-shrinkage-work))

**Ridge: MSE vs. log(Λ)**
![](/images/ridge.png)

**Lasso: MSE vs. log(Λ)**
![](/images/lasso.png)

**Model Summary:**

| Model   | Accuracy   | TPR   |
| -----   | :--------: | :---: |
| **Ridge** | 0.805 | 0.182 |
| **Lasso** | 0.807 | 0.185 |

Based on the model summary, the penalty methods were not that effective.

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
