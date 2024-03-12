# Stock Price Prediction System

## Overview
This system leverages advanced machine learning techniques and algorithms, including TensorFlow and various others, combined with news crawling technologies to predict stock prices. It utilizes data from TradingView and performs analysis based on the latest financial news to make informed predictions.

## Features
- **News Crawling**: Dynamically scrapes the latest financial news for stock analysis.
- **Sentiment Analysis on News**: Extends the news crawling feature by analyzing the sentiment of financial news articles. This involves evaluating the positivity or negativity of news content to assess its potential impact on stock prices.
- **Data Analysis**: Utilizes TensorFlow along with K-Nearest Neighbors (KNN), LSTM models, and other algorithms for predictive analysis.
- **TradingView Integration**: Employs TradingView's charts for displaying stock price movements and predictions.
- **Indicator Analysis**: Incorporates financial indicators such as Bollinger Bands, MACD, RSI, and DEMA for comprehensive analysis.
- **Comprehensive chart view**: Thanks to tradingview the prediction and the actual data is shown in a very familiar way to traders.

## Technologies Used
- **Python**: Primary programming language for backend development.
- **Django**: The web framework for developing the platform's web interface.
- **TensorFlow**: For creating deep learning models.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Yahoo_fin**: To fetch historical stock data.
- **GoogleNews**: For scraping the latest financial news.
- **TextBlob & NLTK**: Used for sentiment analysis on financial news, providing essential tools for text processing and sentiment scoring.
- **TA-Lib**: To compute technical indicators.
- **Sklearn**: For preprocessing and model evaluation.
- **Matplotlib**: (Optional) For future implementation of data visualization.
- **HTML/CSS**: For frontend development.
- **TradingView Charts**: For integrating advanced financial charting capabilities into the platform, enabling dynamic visualization of stock


## Installation
1. Clone the repository:
```
git clone <repository-url>
```
2. Install required Python packages: 
```
pip install -r requirements.txt
```
3. Set up Django:
   ```
   python manage.py makemigrations
   python manage.py migrate
   python manage.py runserver
   ```

## Contributing
Contributions are welcome. Please feel free to fork the repository, make your changes, and submit a pull request if you wish.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Disclaimer
This system is for educational and research purposes only. It is not intended for actual trading. I am not responsible for any financial loss incurred by using this system.

## Acknowledgments
- **TensorFlow Team**: For creating TensorFlow, a vital framework that simplifies deep learning model development.
- **The Python Community**: For their extensive libraries and frameworks, making Python essential in data science and machine learning.
- **Financial Data Providers**: Grateful for the wealth of financial data sources that enrich our analysis and learning.
- **The Django Community**: For developing a web framework that blends simplicity with the power needed for data-centric web projects.
- **Machine Learning Researchers**: For advancing machine learning and financial market prediction, guiding future innovations.

