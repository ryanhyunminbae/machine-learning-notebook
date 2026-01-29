# Machine Learning for Finance: From Fundamentals to Neural Networks

A comprehensive Jupyter notebook that teaches machine learning concepts with hands-on applications to financial markets. This project covers everything from basic ML algorithms to advanced deep learning architectures like LSTMs and Transformers.

## 🎯 What You'll Learn

- **Machine Learning Fundamentals** - Supervised vs unsupervised learning, model evaluation
- **Financial Data Analysis** - Working with stock data, technical indicators, feature engineering
- **Traditional ML Models** - Linear/Logistic Regression, Decision Trees, Random Forests
- **Quantitative Trading Strategies** - Trend-following, pairs trading, multi-factor models
- **Deep Learning** - LSTM and Transformer architectures for time series prediction
- **Risk Management** - Kelly Criterion, portfolio optimization, backtesting
- **Live Trading Dashboard** - Real-time crypto predictions with WebSocket streaming

## 📚 Table of Contents

| Part | Topic | Description |
|------|-------|-------------|
| 1 | Environment Setup | Python essentials and library installation |
| 2 | ML Fundamentals | Core concepts and types of machine learning |
| 3 | Financial Data | Working with stock data using yfinance |
| 4 | Data Preprocessing | Feature scaling, handling missing values, train/test splits |
| 5 | Regression | Predicting continuous values (stock returns) |
| 6 | Classification | Predicting market direction (up/down) |
| 7 | Unsupervised Learning | K-Means clustering, PCA for stocks |
| 8 | Model Evaluation | Cross-validation, avoiding look-ahead bias |
| 9 | Simple Trading Strategy | ML-based trading with backtesting |
| 10 | Advanced Strategies | Trend-following, volatility management |
| 10B | Institutional Strategies | Pairs trading, multi-factor models, portfolio optimization |
| 10C | Neural Networks | LSTM and Transformer architectures |
| 11 | Next Steps | Resources for continued learning |
| 12 | Live Dashboard | Real-time crypto prediction with all models |

## 🛠️ Technologies Used

- **Python 3.9+**
- **PyTorch** - Deep learning (LSTM, Transformers)
- **scikit-learn** - Traditional ML algorithms
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **yfinance** - Stock data download
- **matplotlib/seaborn** - Visualization
- **websocket-client** - Real-time data streaming from Coinbase

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ryanhyunminbae/machine-learning-notebook.git
cd machine-learning-notebook
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook ml_finance_fundamentals.ipynb
```

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
yfinance>=0.2.30
jupyter>=1.0.0
ipykernel>=6.25.0
torch>=2.0.0
scipy>=1.10.0
statsmodels>=0.14.0
websocket-client>=1.6.0
ipywidgets>=8.0.0
ipympl>=0.9.0
```

## 🏗️ Project Structure

```
machine-learning-notebook/
├── ml_finance_fundamentals.ipynb  # Main notebook (7000+ lines)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── *.png                          # Generated visualizations
```

## 📊 Strategies Implemented

### Traditional ML
- **Random Forest Classifier** - Predict market direction
- **Gradient Boosting** - Ensemble learning
- **Logistic Regression** - Binary classification baseline

### Quantitative Strategies
- **Trend Following** - SMA crossover with volatility management
- **Pairs Trading** - Statistical arbitrage on correlated stocks
- **Multi-Factor Alpha Model** - Combining momentum, mean-reversion, volatility signals
- **Portfolio Optimization** - Markowitz mean-variance, Risk Parity

### Deep Learning
- **LSTM Networks** - Sequential pattern recognition
- **Transformer Architecture** - Self-attention for time series

### Risk Management
- **Kelly Criterion** - Optimal position sizing
- **Value at Risk (VaR)** - Downside risk measurement
- **Walk-Forward Validation** - Realistic backtesting

## 🔴 Live Trading Dashboard

The notebook includes a real-time trading dashboard using Coinbase WebSocket:

```python
# In Part 12
rt = RealTimeDashboard('BTC-USD')
rt.run(duration_seconds=300)
```

**Features:**
- Tick-by-tick price updates (no delay)
- All 6 models predicting simultaneously
- Technical indicator visualization (RSI, MACD, SMA)
- Model reasoning - see why models are bullish/bearish
- Performance comparison vs Buy & Hold
- Supports BTC-USD, ETH-USD, SOL-USD, and more

## ⚠️ Disclaimer

**This project is for educational purposes only.** 

- Past performance does not guarantee future results
- The strategies presented are simplified examples
- Real trading involves significant risk of loss
- Always paper trade before risking real capital
- This is not financial advice

## 🎓 Learning Path

1. **Beginner**: Start with Parts 1-6 to understand ML fundamentals
2. **Intermediate**: Parts 7-9 for practical trading applications
3. **Advanced**: Parts 10-12 for institutional-grade strategies, deep learning, and live trading

## 📈 Sample Results

The notebook generates various visualizations including:
- Strategy performance vs buy-and-hold benchmarks
- Feature importance analysis
- Training curves for neural networks
- Risk-return scatter plots
- Drawdown analysis
- Real-time prediction dashboards

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new trading strategies
- Improve documentation

## 📝 License

MIT License - feel free to use this for learning and personal projects.

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for free stock data
- [PyTorch](https://pytorch.org/) for deep learning framework
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Coinbase](https://www.coinbase.com/) for real-time WebSocket API

---

**Happy Learning and Trading!** 📈🤖

*Remember: The best traders are always learning. Markets evolve, and so should your strategies.*
