# 		Stock Prices Predictive Model

​									Project proposal for University of Virginia

​									 SYS 6014 Decision Analysis Spring 2020

​														$Lin\ Wang \\ \quad lw7kv$	

​													   $Mar\ 25,\  2020$

### Introduction

​	As long as capital markets have existed, investors and aspiring arbitrageurs alike have strived to gain edges in predicting stock prices. In particular, use of machine-learning techniques and quantitative analysis to make stock price predictions has become increasingly popular with time. Although for my view, it’s hard and rough to get the accuracy predictive result, because there are definitely many parameters and variables you haven't considered, for the real world, the model is more intricacy and complex. However, it is meaning to do some prediction to avoid the failure and decrease risk. 


<p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325160451231.png">
</p>

​	In these model, by using python, there are several categories of data that can be used when designing a price projection algorithm. These categories and factors summarize a company’s financial history in easy to crunch numbers. These factors are; **sentiment analysis, past prices, sales growth and the dividends** that the company has been paying out to its stockholders. These factors when summarized indicate a company’s vital statistics, and they can be manipulated to predict which circumstances will affect a company’s price in the future and how the company will respond to that. To create a software program that analyzes this data involves installing dependencies, collecting the dataset of the above factors, inputting the script of these factors into the program and finally analyzing the resultant graph.



### Model of the decision problem

- **Support Vector Machine**

​	This method builds the predictive model and graphs it. It takes three parameters: dates, prices, and x (the order of elements). This function creates three models, each of them will be a type of support vector machine. A support vector machine is a linear separator.

​	It takes data that is already classified and tries to predict a set of unclassified data. So, if we only had two data classes it would look like this:

<p align="center">
	<img src = "http://68.media.tumblr.com/0e459c9df3dc85c301ae41db5e058cb8/tumblr_inline_n9xq5hiRsC1rmpjcz.jpg">
</p>


  It will be such that the distances between the closest points in each of the two groups are farthest away. When we add a new data point in our graph depending on which side of the line it is, we could classify it accordingly with the label. However, in this program we are not predicting a class label, so we don't need to classify instead we are predicting the next value in a series which means we want to use regression.

  <p align="center">
      <img src = "http://www.saedsayad.com/images/SVR_1.png">
  </p>


  Support Vector Machine's can be used for regression as well. The support vector regression is a type of SVM that uses the space between data points as a margin of error and predicts the most likely next point in a dataset.

- **Action set $\mathbb{A}$** 

  The general operation of skateholders, such as Climax(Buying/Selling) ; Dumping; Gap ; Liquidation. e.g A climax occurs at the end of a [bull](https://www.investopedia.com/terms/b/bullmarket.asp) or [bear](https://www.investopedia.com/terms/b/bearmarket.asp) market cycle and is characterized by escalated trading volume and sharp price movements. Climaxes are usually preceded by extreme sentiment readings, either excessive euphoria at market peaks, or excessive pessimism at market bottoms.

- **State space $\mathbb{X}$** 

  The financial market, the interest of public companies or specific to say, the captial market. Capital market is a market where buyers and sellers engage in trade of financial securities like bonds, stocks, etc. The buying/selling is undertaken by participants such as individuals and institutions. ... Generally, this market trades mostly in long-term securities.

- **The parameter space $\Theta$** 

  Financial markets are characterized by the uncertainty about the future prices of stocks, currencies, commodities, interest rates or stock indices.

  The set of all possible outcomes is called the Sample Space and we denote it with $\mathbb{X}$ , the probability of a subset $A \in \mathbb{X}$ . We denote it with $P(A)$. 

  

- **The data generation process**

  The initial dataset we used was a dataset used by [Hack/Reduce at the Boston Data Festival Predictive Modeling Hackathon](https://www.kaggle.com/c/boston-data-festival-hackathon/data). The training data consisted of the opening, closing, maximum and minimum prices for 94 stocks over 500 days. The hackathon dataset used stock data from 20 years ago, and we wanted to see how our models would perform on more recent data. We wrote a function that reads in a file with stock tickers, one per row, to create a file of stock price data. We assume that we have a list of the S&P 500 stocks, stored in stocks.csv, that lists each stock ticker, along with the associated company name and industry area. We decide to analyze the S&P 500 stocks in 5 different industry areas.

  1. Consumer Discretionary

  2. Health Care

  3. Information Technology

  4. Financials

  5. Industrials


- **Utility function**  **and** **Loss function** 

  $P$ is a function that assigns numbers to events in the field $\Theta$, $P : \Theta \rightarrow   [0, 1]$ 

  The hinge loss term $\sum_imax(0,1-y_i(w^Tx_i+b))$ in soft margin SVM penalizes *misclassifications*. 

  In hard margin SVM there are, by definition, no misclassifications. 

  This indeed means that hard margin SVM tries to minimize $\|w\|^2$. Due to the formulation of the SVM problem, the margin is $2\over \|w\|$. As such, minimizing the norm of $w$ is geometrically equivalent to maximizing the margin

  

### The predictive model

​	The efficient market hypothesis states that the factors that determine the price of stocks in 	the future are in the future thus making the future prices of stocks random andunpredictable. 	However, using the stock market prediction methods outlined in this program simplifies the 	process of prediction by removing the random element out of stock market price futures. The 	use of support vector machines for classification and regression analyses gives a scientific 	 	element to the prediction of stock market prices rather than relying on hunches and intuition. 	A combination of the thus computed results and sound investment planning will, therefore, 	raise an investor’s chance of stock market success.

- **RBF(Radial basis function)**

  Radial basis function (RBF) networks typically have three layers: an input layer, a hidden layer with a non-linear RBF activation function and a linear output layer. We use RBF model as standard to fit our data, and as a standard to test the performance of predictive model.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Radial_funktion_network.svg/1920px-Radial_funktion_network.svg.png" alt="img" style="zoom:20%;" />

​		Out funtion $\varphi(x)=\sum^N_{i=1}a_i\rho(\|x-c_i\|)$ 

​		Basis function centers can be randomly sampled among the input instances or obtained by 	    Orthogonal Least Square Learning Algorithm or found by [clustering](https://en.wikipedia.org/wiki/Data_clustering) the samples and		 		choosing the cluster means as the centers.

​		The RBF widths are usually all fixed to same value which is proportional to the maximum  		distance between the chosen centers.



- **Polynomial Functions model**

  A [polynomial function](https://en.wikipedia.org/wiki/Polynomial_function) is one that has the form

  $y=a_nx^n+a_{n-1}x^{n-1}+...+a_2x^2+a_1x^1+a_0$

  where *n* is a non-negative [integer](https://en.wikipedia.org/wiki/Integer) that defines the degree of the polynomial. A polynomial with a degree of 0 is simply a [constant function](https://en.wikipedia.org/wiki/Constant_function); with a degree of 1 is a [line](https://en.wikipedia.org/wiki/Linear_function); with a degree of 2 is a [quadratic](https://en.wikipedia.org/wiki/Quadratic_function); with a degree of 3 is a [cubic](https://en.wikipedia.org/wiki/Cubic_function), and so on.

  
<p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325162624600.png">
</p>



  Historically, polynomial models are among the most frequently used empirical models for [curve fitting](https://en.wikipedia.org/wiki/Curve_fitting).

- **SVR**

  SVR gives us the flexibility to define how much error is acceptable in our model and will find an appropriate line (or hyperplane in higher dimensions) to fit the data.

  In contrast to OLS, the objective function of SVR is to minimize the coefficients — more specifically, the $l_2-norm$ of the coefficient vector — not the squared error. The error term is instead handled in the constraints, where we set the absolute error less than or equal to a specified margin, called the maximum error, $\varepsilon$ 

  objective function: ${Min} {1\over2}\|w\|^2+C\sum^n_{i=1}|\xi_i|$ 

  constraints: $|y_i-w_ix_i|≤$ $\varepsilon+|\xi_i|$ 

  where yᵢ is the target, wᵢ is the coefficient, xᵢ is the predictor and for any value that falls outside of  $\varepsilon$ , we can denote its deviation from the margin as $\xi$.

 <p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325154907362.png">
</p>

  The plot below shows the results of a trained SVR model on the Boston Data Festival Hackathon data. The red line represents the line of best fit and the black lines represent the margin of error, ϵ, which we set to 5 ($5,000) and set *C*=1.0.


<p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325155112426.png">
</p>

  The above model seems to fit the data much better. We can go one step further and grid search over *C* to obtain an even better solution. Let’s define a scoring metric, $\% within Epsilon$ This metric measures how many of the total points within our test set fall within our margin of error. We can also monitor how the Mean Absolute Error (*MAE*) varies with *C* as well.

  Below is a plot of the grid search results, with values of *C* on the x-axis and *% within Epsilon* and *MAE* on the left and right y-axes, respectively.


 <p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325155514433.png">
</p>


  As we can see, *MAE* generally decreases as *C* increases. However, we see a maximum occur in the *% within Epsilon* metric. Since our original objective of this model was to maximize the prediction within our margin of error ($5,000), we want t find the value of *C* that maximizes *% within Epsilon*. Thus, *C*=6.13.

  Let’s build one last model with our final hyperparameters, ϵ=5, *C*=6.13.

 <p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325155547750.png">
</p>

  The plot above shows that this model has again improved upon previous ones, as expected. 

  

### Quantifying the value-add of your tool

​	Generally, after executing the program, we will get a combination graphs of RBF model, 	 	linear model, polynomial model and original data. We can check the bias and error directly  	from the plots. 

​	For RBF : $a_i(t+1)=a_i(t)+v[x(t+1)-\varphi((t),w)]{u(\|x(t)-c_i)\over \sum^N_{i=1}u^2(\|x(t)-c_i\|)}$

​	For polynomial:  $y=a_nx^n+a_{n-1}x^{n-1}+...+a_2x^2+a_1x^1+a_0$

​	SVR:  ${Min} {1\over2}\|w\|^2+C\sum^n_{i=1}|\xi_i|$

- **Dependencies**

  The dependencies that are installed in the program need to enable the user to collect the dataset with ease, calculate and interpret the numbers in the dataset, build a predictive model based on the past dataset and build a projective model for the future of the stock prices. When running in synchrony, the dependencies help in developing a support vector machine. A support vector machine primarily is a linear separator that takes data that is classified and attempts to predict and classify unclassified data. The support vector machine aid in the calculation of the support vector regression which can be calculated to accurately determine how each addition of data or alteration of market factors will alter the price of stocks. The four dependencies include:

  `pip install csv` : To read data from the stock prices [csv](https://pypi.python.org/pypi/csv)

  `pip install numpy` : To perform calculations  [numpy](http://www.numpy.org/)

  `pip install scikit-learn` : To build a predictive model [scikit-learn](http://scikit-learn.org/)

  `pip install matplotlib` : To plot datapoints on the model to analyze [matplotlib](http://matplotlib.org/)

  

### Results & Conclusion 

<p align="center">
	<img src = "https://github.com/UVA-Engineering-Decision-Analysis/Lin_Wang-Project-SYS-6014/blob/master/image-20200325151321322.png">
</p>
​	

​	On analyzing the graph, we see that each of our models shows up in the graph and the RBF 	model seems to fit our data the best. Hence, we can use it's prediction in the command line to 	make stock predictions

​	The support vector regression estimates how each addition or modification of data affects the 	prediction and outlook on the future prices of stock. The support vector regression can be 	 	developed by using either the linear function model, the polynomial functions model or the 	ration basis model. The different results can then be plotted on one or different graphs for 	 	analysis). These graphs are then compared with the actual data from the company’s history 	and the model that matches the historical data and trends can then be used to predict how 	the figures will react to market stimuliate.



### Reference 

- [Introduction to Support Vector Machines](https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html#introduction-to-support-vector-machines)
- [Stock Market Prediction Using](https://nicholastsmith.wordpress.com/2016/04/20/stock-market-prediction-using-multi-layer-perceptrons-with-tensorflow/)
- [Stock Price Prediction With Big Data and Machine Learning](http://eugenezhulenev.com/blog/2014/11/14/stock-price-prediction-with-big-data-and-machine-learning/)
- [How I made $500k with machine learning and HFT (high frequency trading)](https://jspauld.com/post/35126549635/how-i-made-500k-with-machine-learning-and-hft)
- [Boston Data Festival Hackathon](https://www.kaggle.com/c/boston-data-festival-hackathon/data)
- [Buffalo Capital Management](https://sites.google.com/site/predictingstockmovement/dataset-descriptions)
