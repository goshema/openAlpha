# OpenAlpha: Institutional Multi-Factor Equity Screener

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

**OpenAlpha** is a production-grade, fully open-source quantitative equity screening tool designed for long-term (3Y+) institutional portfolio construction. It bypasses retail-level heuristics in favor of robust statistical normalization, advanced risk modeling, and sector-neutral factor aggregation.

Built entirely with open-source libraries, it requires no proprietary data feeds or closed-source APIs.

---

## ️ Architecture & Methodology

Unlike standard screeners that simply sort by a single metric, OpenAlpha constructs a **Composite Factor Score (0-100)** based on five distinct, uncorrelated investment pillars. 

### 1. The Five Factor Pillars
Every stock in the universe is evaluated across five dimensions. Each dimension is composed of multiple sub-metrics, normalized, and aggregated:

*   **Quality (30% Weight):** Measures profitability, efficiency, and earnings quality.
    *   *Metrics:* ROIC, Gross Margin, Operating Margin, FCF Margin, Revenue/Earnings Growth, Accruals (inverted).
*   **Value (25% Weight):** Measures relative cheapness compared to sector peers.
    *   *Metrics:* Sector-Relative P/E, Price-to-Book, EV/EBITDA. *(Note: Lower values yield higher scores).*
*   **Momentum (20% Weight):** Captures long-term price trends and relative strength.
    *   *Metrics:* 12-1 Month Momentum, Benchmark-Adjusted Momentum, 52-Week High Proximity, 3-Year Relative Strength, RS Trend.
*   **Low Risk (15% Weight):** Identifies stocks with favorable risk profiles.
    *   *Metrics:* Vasicek-Adjusted Beta, Downside Beta, Idiosyncratic Volatility, CVaR (95%), Max Drawdown, Ulcer Index. *(Note: Lower risk yields higher scores).*
*   **Sentiment (10% Weight):** Incorporates forward-looking analyst and insider data.
    *   *Metrics:* Analyst Target Upside, Recommendation Score, Insider Ownership.

### 2. Robust Normalization (The "Secret Sauce")
Standard z-scores are easily ruined by outliers (e.g., a single tech giant with a massive P/E ratio). OpenAlpha uses **Median Absolute Deviation (MAD)** scaling to ensure robustness.

For every metric $X$, the robust z-score is calculated as:

$$ Z = \frac{X - \text{Median}(X)}{\text{MAD}(X) \times 1.4826} $$

Where $1.4826$ is the consistency constant for a normal distribution. Finally, all z-scores are **winsorized (clipped) to the range $[-3, +3]$** to prevent any single outlier from dominating the composite score.

### 3. Sector Neutrality
To prevent the screener from simply picking the entire Technology sector when tech is booming, normalization is performed **within each sector**. A stock is only compared to its sector peers. If the universe is too small (<30 stocks), it gracefully falls back to global scaling.

---

## 📐 Mathematical Deep Dive

### Vasicek Beta Shrinkage
Raw historical beta is a noisy estimator. To reduce estimation error, we apply the Vasicek shrinkage technique, which pulls the raw beta toward the market average (1.0):

$$ \beta_{adjusted} = 0.67 \times \beta_{raw} + 0.33 \times 1.0 $$

### Idiosyncratic Volatility
We isolate stock-specific risk by calculating the residuals of the CAPM regression. This is the volatility that cannot be explained by the broader market (SPY):

$$ \sigma_{idio} = \sqrt{\text{Var}(\epsilon)} \times \sqrt{252} $$
*Where $\epsilon = R_{stock} - (\alpha + \beta_{raw} R_{market})$*

### Ulcer Index
Unlike standard deviation, which penalizes upside volatility, the Ulcer Index measures the depth and duration of drawdowns from recent highs:

$$ \text{Ulcer} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} DD_i^2} $$
*Where $DD_i$ is the percentage drawdown from the running maximum at time $i$.*

### Macro Stress Overlay
The model includes a built-in regime filter. If the benchmark (SPY) drops more than 1.5% in a day, or if the VIX exceeds 30, the composite scores of all equities are penalized by 20% to reflect heightened systemic risk.

---

## 📦 Portfolio Construction

Once the top stocks are identified, OpenAlpha constructs a model portfolio using institutional constraints:

1.  **Sector Diversification:** The top 10 stocks are selected, but constrained to a maximum of **2 stocks per sector**.
2.  **Inverse-Volatility Weighting:** Instead of equal weighting, capital is allocated inversely to the stock's idiosyncratic volatility. This ensures that riskier stocks receive smaller allocations, stabilizing the overall portfolio variance:
    $$ w_i = \frac{1 / \sigma_{idio, i}}{\sum_{j=1}^{N} (1 / \sigma_{idio, j})} $$

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.11 or higher
*   An active internet connection (for Yahoo Finance data)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/goshema/open-alpha.git
   cd open-alpha
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually: `pip install streamlit pandas numpy yfinance plotly requests streamlit-autorefresh`)*

### Run the App
```bash
streamlit run screener.v5.py
```
The application will open automatically in your default web browser at `localhost:8501`.

---

## ⚙️ Features

*   **Batch Data Pipeline:** Downloads 500+ tickers in a single HTTP request, reducing fetch time from 3+ minutes to ~15 seconds and preventing IP rate-limiting.
*   **Look-Ahead Bias Prevention:** Uses strictly trailing/realized data for fundamental metrics.
*   **Non-Blocking UI:** Utilizes `streamlit-autorefresh` to update data in the background without freezing the user interface.
*   **Deep Dive Analytics:** Interactive Plotly charts for Price/Relative Strength, Factor Heatmaps, and Risk/Reward scatter plots.
*   **Three-Tier Fallback:** If Wikipedia fails to load the S&P 500 list, it automatically falls back to a GitHub CSV, and finally to a hardcoded list of mega-cap stocks.

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice, investment recommendations, or an offer to buy or sell any securities. 

Quantitative models are based on historical data, which is not indicative of future results. The author assumes no liability for any financial losses incurred through the use of this tool. Always conduct your own due diligence and consult with a licensed financial advisor before making investment decisions.

---

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/goshema/open-alpha-screener/issues).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
