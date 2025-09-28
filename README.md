# Sentiment_Aided_Returns_Distribution
# 📈 Returns Distribution Analysis 
## *AI-Powered Market Prediction Using Sentiment & Technical Fusion*

**Status:** Active Development | **Language:** Python 3.8+ | **Framework:** TensorFlow 2.0+ | 

*Revolutionizing quantitative finance through multi-modal deep learning*

---

##  **Project Overview**

A cutting-edge financial machine learning system that predicts next-day market returns by intelligently fusing **technical analysis** with **news sentiment analysis**. Our dual-CNN architecture processes 44 engineered technical features alongside multi-dimensional news sentiment scores across 5 hierarchical levels (Company → Group → Sector → Nation → Global).

### ** Key Innovation**
- **Multi-Modal Fusion**: First-of-its-kind CNN architecture that treats technical indicators and sentiment data as "financial images"
- **Hierarchical Sentiment Analysis**: 5-tier sentiment scoring system capturing micro to macro market impacts
- **Advanced Feature Engineering**: 44 sophisticated technical indicators across 8 specialized modules
- **Real-time Scalability**: Designed for production-grade algorithmic trading systems

---

## 📊 **Architecture Overview**

### **Data Flow Pipeline**
```
OHLCV Data (2008-Present) → Technical Analysis Module → 44 Features → Technical CNN → p_tech ∈ R¹¹
                                                                            ↘
News Data (24 Categories) → Sentiment Analysis Module → 5-Tier Matrix → Sentiment CNN → p_news ∈ R¹¹
                                                                            ↗
                                                      Fusion Module → p_final ∈ R¹⁵
```

### **🔧 Technical Analysis Module (44 Features)**

#### **1. Trend Analysis (5 features)**
- **Trend Direction**: Bullish(+1)/Neutral(0)/Bearish(-1) classification
- **Trend Strength**: ADX-based strength scoring (0-10 scale)
- **Trend Smoothness**: Price deviation from HT_TRENDLINE
- **Trend Persistence**: SAR consistency measurement
- **Trend Confidence**: Multi-indicator confirmation percentage

#### **2. Momentum Analysis (5 features)**
- **Momentum Score**: Weighted combination of ROC, MACD, RSI, CCI (-10 to +10)
- **Momentum Acceleration**: Rate of momentum change
- **Overbought/Oversold State**: Multi-oscillator extreme condition detection
- **Divergence Flag**: Price vs momentum trend comparison
- **Exhaustion Risk**: Intensity of extreme market conditions

#### **3. Oscillator Analysis (3 features)**
- **OB/OS Score**: Aggregated oscillator extremes (0-10)
- **Reversion Bias**: Mean reversion probability (-1 to +1)
- **Signal Freshness**: Recency of current market state (0-1)

#### **4. Volatility Analysis (5 features)**
- **Volatility Level**: ATR percentile ranking (0-10)
- **Volatility Regime**: Low/Medium/High classification
- **Volatility Squeeze**: Compression detection flag
- **Volatility Trend**: Direction of volatility change
- **Shock Risk**: Sudden expansion probability (0-10)

#### **5. Bollinger Bands Analysis (4 features)**
- **Band Position**: Price location within bands (0-100)
- **Band Walk**: Edge movement intensity (0-10)
- **Squeeze Intensity**: Band compression measurement
- **Breakout Likelihood**: Expansion probability estimate

#### **6. Cycle Analysis (3 features)**
- **Cycle Stability**: HT_DCPERIOD consistency (0-10)
- **Adaptive Bias**: KAMA directional movement
- **Market Choppiness**: SAR flip frequency measurement

#### **7. Strength Analysis (3 features)**
- **Strength Index**: ADX-derived market strength
- **Strength Stability**: ADX deviation measurement
- **Tradability Flag**: Systematic trading condition indicator

#### **8. Risk Analysis (2 features)**
- **Systematic Risk Level**: Beta-scaled market correlation
- **Directional Coupling**: Beta deviation normalization

---

## 🗞️ **Sentiment Analysis Module**

### **5-Tier Hierarchical Sentiment Scoring**

#### ** Company Sentiment**
Direct impact on individual stock prices
- **Examples**: "Infosys wins $2.5B Danske Bank deal", "Paytm banned by RBI"
- **Scope**: Earnings, governance, regulatory actions, product launches

#### ** Group Sentiment**  
Conglomerate-level impact across subsidiaries
- **Examples**: "Adani Green secures $1.4B funding", "Hindenburg Research report"
- **Scope**: Financing events, reputational shocks, strategic partnerships

#### ** Sector Sentiment**
Industry-wide systematic impacts
- **Examples**: "RBI increases banking risk weights", "PLI scheme for pharma"
- **Scope**: Regulatory reforms, commodity cycles, policy incentives

#### ** Nation Sentiment**
Broad Indian market influences
- **Examples**: "GDP growth hits 8.2%", "CPI inflation rises to 7.4%"
- **Scope**: Macroeconomic data, fiscal policy, political events

#### ** Global Sentiment**
International spillover effects
- **Examples**: "Fed pivots to rate cuts", "Silicon Valley Bank collapse"
- **Scope**: Global monetary policy, commodity cycles, geopolitical risks

### **Data Sources & Categories (24 Types)**
```
Commodity News | Prices | Company News | Industry Announcements
Inflation | Market Pulse | Closing Bell | Corporate News
Global Markets | Mid Day Review | Movers Today | Equity News
Opening Bell | F&O Corner | IPO Analysis | IPO News
Money Markets | Call Money | Forex Rates | Forex Reserves
Finance | Politics & Current Affairs | And more...
```

---

##  **Deep Learning Architecture**

### **Technical CNN Module**
```python
Input: Technical Features Matrix (44 indicators × time_window)
↓
Conv1D layers with feature extraction
↓
Pooling and normalization
↓
Dense layers with dropout
↓
Output: p_tech ∈ R¹¹ (probability distribution)
```

### **Sentiment CNN Module**
```python
Input: Sentiment Matrix (N×S) where N=companies, S=sentiment_types
↓
2D CNN for spatial correlation learning
↓
Multi-channel processing for category fusion
↓
Flatten and dense layers
↓
Output: p_news ∈ R¹¹ (probability distribution)
```

### **Fusion Strategies**

#### **1. Concatenation + MLP (Meta-Model)**
```python
x = [p_tech ∥ p_news] ∈ R²²
→ MLP with ReLU + Dropout
→ Softmax → p_final ∈ R¹⁵
```

#### **2. Weighted Averaging**
```python
p_final = α × p_tech + (1-α) × p_news
# α can be fixed or dynamic via gating function
```

#### **3. Mixture of Experts (Dynamic)**
```python
[w_tech, w_news] = GatingNetwork(regime_features)
p_final = w_tech × p_tech + w_news × p_news
```

---

## 📈 **Market Coverage & Data**

### **Universe**: Nifty-50 Companies
### **Timeframe**: 2008-Present (Daily)
### **Data Volume**: 
- **OHLCV Records**: 15+ years × 50 companies × 365 days ≈ 273,750 records
- **News Articles**: 100,000+ categorized news items
- **Technical Features**: 44 indicators × daily computation
- **Sentiment Scores**: 5 levels × 24 categories × daily aggregation

---

## 🛠️ **Technical Implementation**

### **Key Mathematical Formulations**

#### **Moving Averages**
```
SMA = (1/n) × Σᵢ₌₁ⁿ Pᵢ
EMA_t = α × P_t + (1-α) × EMA_{t-1}, where α = 2/(n+1)
```

#### **Momentum Indicators**
```
RSI = 100 - (100/(1+RS)), RS = Average_Gain/Average_Loss
MACD = EMA₁₂ - EMA₂₆
ROC = ((P_t - P_{t-n})/P_{t-n}) × 100
```

#### **Volatility Measures**
```
Bollinger Bands: UB = SMA + k×σ, LB = SMA - k×σ
Williams %R = ((H_n - C)/(H_n - L_n)) × (-100)
```

### **Dependencies**
```python
tensorflow>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
ta-lib>=0.4.0
yfinance>=0.1.70
beautifulsoup4>=4.9.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

