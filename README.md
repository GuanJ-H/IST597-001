## Term Project of IST 597-001: Stock Market Prediction
Stock market prediction is an very important and classic topic in financial economics. A good prediction of a stock's future movement can provide insight about the market behavior over time, and then bring significant profit. With the increasingly computational power
of the computer, machine learning and deep learning techniques become efficient approaches to solve this problem.

### Goals
The goal of this project is to layout deep investment techniques in financial markets using  artificial neural networks (e.g. deep learning models). Stock market prediction always involve variety of data which is difficult to design an ideal economic model. Deep learning models are able to exploit potentiallt non-linear patterns in such data which can help prediction.

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


