# wine-quality-machine-learning-example
Using linear regression to fit a model that predicts the score of wine based on physiochemical tests. Below are the output plots which show that linear relationship between wine quality and the two most influential variables for that respective wine.

![alt text](https://raw.githubusercontent.com/Mephastophilis/wine-quality-machine-learning-example/master/red_wine_plot.png)

![alt text](https://raw.githubusercontent.com/Mephastophilis/wine-quality-machine-learning-example/master/white_wine_plot.png)

The code also performs regularized linear regression and uses SVM to fit the data. It also explores using a multi-layer percepton neural network for classifying the wine to the scores of 1 to 10. It tries various hidden layer sizes and finds which of those produces a classifier with the highest F-1 score.

This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib
