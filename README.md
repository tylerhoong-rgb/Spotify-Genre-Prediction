## Spotify Genre Prediction

Data from https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres

This project is a full implementation of a multi-class classification models on raw song lyric data. \
I started with a preprocessing of over 550,000 songs, including computing a two-stage hybrid filter apporach with Jaccard similarity score to detect and remove duplicate song titles.\
I then applied unigram tokenization and computed TF-IDF scores to create the features. \
I created two models: a baseline complement Naive Bayes and a Logistic Regression tuned with 5-fold stratified CV.

A website where you can test the model by looking up a song or pasting lyrics is at: https://tylerhoong-rgb.github.io/Spotify-Genre-Prediction/

## Results, Takeaways, and Questions for Future Study
Naive Bayes Roc_Auc: 81.3%
Logistic Regression Roc_Auc: 84.3%

Covers are songs sung by another artist than the original, changing the style and sometimes genre of the song. A way to deal with this for the data would be to edit the similarity detector to not filter by artist and remove covers. However, this would not actually solve the issue, just remove it from the dataset used for the model. Training on prosodic features of the song lyrics would help differentiate covers and solve this issue, but this data is not available in this dataset.

There is another variable in the data, `niche_genre`, that is more varied than `genre`. It is also treated as a list in the data, so 1 song can have multiple of these sub-genres. This would make for an interesting follow-up project.

Duplicate songs were tracked with a similarity score approach with an arbitrary 90% similarity baseline. Tuning this baseline or finding other approaches such as a complex regex expression for the title or unsupervised clustering could improve how many duplicates are found.

#### Other Models

LSTMs and MLPs are known to work well for text classification. However, I could not implement an MLP due to compute limits. LSTMs are slower to train and would warrant an alternative tokenization. 
