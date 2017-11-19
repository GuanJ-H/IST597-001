## Term Project of IST 597-001: Stock Market Prediction
Stock market prediction is an very important and classic topic in financial economics. A good prediction of a stock's future movement can provide insight about the market behavior over time, and then bring significant profit. With the increasingly computational power
of the computer, machine learning and deep learning techniques become efficient approaches to solve this problem.

Stock market prediction tends to be a very complicated and difficult problem for several reasons. The most obvious factor is the sheer number of complex variables involved in stock price changes like political change and news coverage [5]. Another issue for stock prediction is the Efficient Market Hypothesis (EMH). In 1965, Eugene F. Fama proposed that the stock market price follows a random-walk pattern where  stocks compete in a zero-sum game with a roughly 50% chance to gain or lose at each point in time [6]. Stock market prediction remains a difficult problem with significant opportunities for improvement through the application of deep learning models. 

### Goals
The goal of this project is to layout deep investment techniques in financial markets using  artificial neural networks (e.g. deep learning models). Stock market prediction always involve variety of data which is difficult to design an ideal economic model. Deep learning models are able to exploit potentially non-linear patterns in such data which can help prediction.

### Dataset
Dataset Name: Yahoo!finance<br />
Data Attibutes:
&nbsp;Open, High, Low, Close, Adjusted Close, Volume, Time

### Basic Approach
Since the stock market data are actually time-series data, recurrent neural networks (RNNs) can be applied to predict stock market movement. RNN is one type of artificial neural networks that take advantages of the the sequential nature of time-series data. Some other traditional methods (e.g. Logistic Regression, Support Vector Machine, etc.) will also be implemented as baseline models. In addition, if time permits, we will also try sentiment/textual analysis and see its influence on stock market prediction.

### Anticipated Results
Our proposed approach will be compared with the baseline model in terms of loss, accuracy, receiver operating characteristic (ROC). We anticipate it can take advantages of the sequential nature and outperform the baseline models.

### Group Members
Jin Zhang: Developer <br />
Jacob Oury: Developer<br />
Guanjie Huang: Developer

### Relevant Bibliography
[1] Chong, Eunsuk, Chulwoo Han, and Frank C. Park. "Deep learning networks for stock market analysis and prediction: Methodology, data representations, and case studies." Expert Systems with Applications 83 (2017): 187-205.<br />
[2] Aggarwal, Saurabh, and Somya Aggarwal. "Deep Investment in Financial Markets using Deep Learning Models." International Journal of Computer Applications 162.2 (2017).<br />
[3] Ding, Xiao, et al. "Deep Learning for Event-Driven Stock Prediction." Ijcai. 2015.<br />
[4] Wikipedia contributors. "Stock market prediction." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 15 Sep. 2017. Web. 16 Nov. 2017.<br />
[5] Lgkvist, M., Karlsson, L., & Loutfi, A. (2014). A review of unsupervised feature learning and deep learning for time-series modeling. Pattern Recognition Letters, 42(1), 11-24. https://doi.org/10.1016/j.patrec.2014.01.008<br />
[6] Fama, E. F. (1965). The Behavior of Stock-Market Prices. The Journal of Business, 38(1), 34-105. https://doi.org/10.1017/CBO9781107415324.004<br />
Lgkvist, M., Karlsson, L., & Loutfi, A. (2014). A review of unsupervised feature learning and deep learning for time-series modeling. Pattern Recognition Letters, 42(1), 11-24. https://doi.org/10.1016/j.patrec.2014.01.008<br />


