# pip install streamlit yfinance pandas numpy plotly ta scikit-learn textblob

# nse_stock_prediction_app_fixed.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NSE AI Stock Predictor - Auto-Refresh Trading",
    page_icon="🇮🇳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 5px 0 0 0;
        font-size: 0.8rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .status-active {
        background-color: #28a745;
        box-shadow: 0 0 8px #28a745;
    }
    
    .status-inactive {
        background-color: #dc3545;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }
    .metric-card h3 {
        margin: 0 0 5px 0;
        font-size: 0.75rem;
        color: #555;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 0.75rem;
        font-weight: bold;
        color: #1e3c72;
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, #00a86b, #008f5f);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: white;
    }
    .signal-buy {
        background: linear-gradient(135deg, #28a745, #208637);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: white;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #6c757d, #5a6268);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: white;
    }
    .signal-sell {
        background: linear-gradient(135deg, #dc3545, #c82333);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: white;
    }
    .signal-strong-sell {
        background: linear-gradient(135deg, #bd2130, #a71d2a);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: white;
    }
    
    .info-box {
        background: rgba(255,255,255,0.95);
        border-left: 3px solid #007bff;
        padding: 8px 12px;
        border-radius: 5px;
        margin: 8px 0;
        font-size: 0.8rem;
    }
    
    .refresh-toggle {
        background: rgba(255,255,255,0.9);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .small-text {
        font-size: 1.8rem;
        color: #666;
    }
    
    .update-time {
        font-size: 0.7rem;
        color: #888;
        text-align: right;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AutoRefreshStockPredictor:
    def __init__(self, symbol, interval='5m', period='5d'):
        self.symbol = self.format_nse_symbol(symbol)
        self.interval = interval
        self.period = period
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.last_update = None
        self.prediction_history = []
        
    @staticmethod
    def format_nse_symbol(symbol):
        symbol = symbol.upper().strip()
        symbol = symbol.replace('.NS', '')
        return f"{symbol}.NS"
    
    def fetch_data(self):
        """Fetch real-time NSE stock data"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval=self.interval)
            if not self.data.empty:
                self.data['Currency'] = '₹'
            return self.data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self):
        """Add technical indicators"""
        if self.data is None or self.data.empty:
            return None
        
        df = self.data.copy()
        
        # Moving Averages
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Money_Flow_Index'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Price features
        df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Support/Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Target
        df['Target'] = df['Close'].shift(-3) > df['Close']
        
        return df.dropna()
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        feature_columns = [
            'SMA_10', 'SMA_20', 'EMA_9', 'EMA_21', 'RSI',
            'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
            'Volume_Ratio', 'Money_Flow_Index', 'High_Low_pct',
            'Price_Change', 'Volatility', 'ATR'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features]
        y = df['Target'] if 'Target' in df.columns else None
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, available_features
    
    def train_model(self):
        """Train the ML model"""
        df = self.add_technical_indicators()
        if df is None or len(df) < 100:
            return None
        
        X, y, features = self.prepare_features(df)
        
        if y is None or len(y) < 50:
            return None
        
        # Ensemble model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        ensemble = VotingRegressor([('rf', rf_model), ('gb', gb_model)])
        
        ensemble.fit(X, y)
        
        rf_model.fit(X, y)
        self.feature_importance = dict(zip(features, rf_model.feature_importances_))
        
        self.model = {
            'ensemble': ensemble,
            'last_X': X[-1:] if len(X) > 0 else None,
            'features': features
        }
        
        self.last_update = datetime.now()
        return self.model
    
    def predict(self):
        """Generate prediction"""
        if self.model is None:
            return None
        
        df = self.add_technical_indicators()
        if df is None:
            return None
        
        X, _, _ = self.prepare_features(df)
        if len(X) == 0:
            return None
        
        latest_features = X[-1:]
        pred = self.model['ensemble'].predict(latest_features)[0]
        
        probability = pred * 100
        confidence = abs(pred - 0.5) * 200
        
        # Price targets
        latest_price = df['Close'].iloc[-1]
        support = df['Support'].iloc[-1] if 'Support' in df.columns else latest_price * 0.98
        resistance = df['Resistance'].iloc[-1] if 'Resistance' in df.columns else latest_price * 1.02
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else latest_price * 0.02
        
        prediction_result = {
            'prediction': 'Bullish' if pred > 0.5 else 'Bearish',
            'probability': min(probability, 100),
            'confidence': min(confidence, 100),
            'raw_score': pred,
            'current_price': latest_price,
            'support': support,
            'resistance': resistance,
            'atr': atr,
            'timestamp': datetime.now()
        }
        
        # Store history
        self.prediction_history.append(prediction_result)
        if len(self.prediction_history) > 20:
            self.prediction_history.pop(0)
        
        return prediction_result
    
    def get_prediction_stability(self):
        """Check if predictions are stable or changing rapidly"""
        if len(self.prediction_history) < 3:
            return "Unstable"
        
        recent_predictions = [p['prediction'] for p in self.prediction_history[-5:]]
        if len(set(recent_predictions)) == 1:
            return "Stable ✓"
        elif len(set(recent_predictions)) > 2:
            return "Volatile - Wait ⚠️"
        else:
            return "Moderate - Monitor 📊"

def plot_chart(data, symbol):
    """Create trading chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{symbol} - Price Chart', 'RSI', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'EMA_9' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['EMA_9'], 
                      name='EMA 9', line=dict(color='cyan', width=1)),
            row=1, col=1
        )
    
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], 
                      name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    if 'Volume' in data.columns:
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                  for i in range(len(data))]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                   marker_color=colors, showlegend=False),
            row=3, col=1
        )
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def display_recommendations(predictions):
    """Display trading recommendations"""
    if not predictions:
        return
    
    confidence = predictions['confidence']
    prediction = predictions['prediction']
    
    # Determine signal
    if prediction == 'Bullish':
        if confidence > 70:
            signal_class = "signal-strong-buy"
            signal_text = "🔥 STRONG BUY SIGNAL"
        elif confidence > 50:
            signal_class = "signal-buy"
            signal_text = "📈 BUY SIGNAL"
        else:
            signal_class = "signal-neutral"
            signal_text = "🔄 NEUTRAL SIGNAL"
    else:
        if confidence > 70:
            signal_class = "signal-strong-sell"
            signal_text = "⚠️ STRONG SELL SIGNAL"
        elif confidence > 50:
            signal_class = "signal-sell"
            signal_text = "📉 SELL SIGNAL"
        else:
            signal_class = "signal-neutral"
            signal_text = "🔄 NEUTRAL SIGNAL"
    
    # Main signal box
    st.markdown(f"""
    <div class="{signal_class}" style="margin-bottom: 15px;">
        <h2 style="margin: 0; font-size: 1.3rem;">{signal_text}</h2>
        <p style="margin: 5px 0 0 0;">Confidence: {predictions['confidence']:.0f}% | Probability: {predictions['probability']:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trading plan
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>💰 ENTRY</h3>
    #         <h2>₹{predictions['current_price']:.2f}</h2>
    #         <small>Current Price</small>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col2:
    #     if prediction == 'Bullish':
    #         stop_loss = predictions['current_price'] - predictions['atr'] * 1.5
    #         st.markdown(f"""
    #         <div class="metric-card">
    #             <h3>🛑 STOP LOSS</h3>
    #             <h2 style="color: #dc3545;">₹{stop_loss:.2f}</h2>
    #             <small>{((stop_loss - predictions['current_price'])/predictions['current_price']*100):.1f}%</small>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     else:
    #         stop_loss = predictions['current_price'] + predictions['atr'] * 1.5
    #         st.markdown(f"""
    #         <div class="metric-card">
    #             <h3>🛑 STOP LOSS</h3>
    #             <h2 style="color: #dc3545;">₹{stop_loss:.2f}</h2>
    #             <small>+{((stop_loss - predictions['current_price'])/predictions['current_price']*100):.1f}%</small>
    #         </div>
    #         """, unsafe_allow_html=True)
    
    # with col3:
    #     if prediction == 'Bullish':
    #         target = predictions['current_price'] + predictions['atr'] * 2.5
    #         st.markdown(f"""
    #         <div class="metric-card">
    #             <h3>🎯 TARGET</h3>
    #             <h2>₹{target:.2f}</h2>
    #             <small>+{((target - predictions['current_price'])/predictions['current_price']*100):.1f}%</small>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     else:
    #         target = predictions['current_price'] - predictions['atr'] * 2.5
    #         st.markdown(f"""
    #         <div class="metric-card">
    #             <h3>🎯 TARGET</h3>
    #             <h2>₹{target:.2f}</h2>
    #             <small>{((target - predictions['current_price'])/predictions['current_price']*100):.1f}%</small>
    #         </div>
    #         """, unsafe_allow_html=True)
    
    # Key levels
    st.markdown(f"""
    <div class="small-text" style="text-align: left; margin-top: 10px;">
        📊 Support: ₹{predictions['support']:.2f} | Resistance: ₹{predictions['resistance']:.2f} | ATR: ₹{predictions['atr']:.2f}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>AI Stock Predictor for Intraday Trading</h1>
        <p>Auto-updates every 30 seconds | Real-time AI Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
        st.session_state.model_trained = False
        st.session_state.predictions = None
        st.session_state.last_update = None
        st.session_state.auto_refresh_counter = 0
        st.session_state.last_train_time = None
        st.session_state.rerun_counter = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        symbol = st.text_input("NSE Stock Symbol", value="RELIANCE").upper()
        
        interval = st.selectbox("Time Interval", ['1m', '5m', '15m', '30m', '1h'], index=1)
        period = st.selectbox("Data Period", ['1d', '5d', '1mo', '3mo'], index=1)        
        
        # Manual train button
        manual_train = st.button("🚀 Get Recommendation", width='stretch')

        # Auto-refresh controls
        st.markdown("## 🔄 Auto-Refresh Settings")
        
        auto_refresh = st.checkbox("Enable Auto-Refresh (30 seconds)", value=True)
        
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 15, 60, 30, step=5)
            st.info(f"🟢 Model will retrain every {refresh_interval} seconds")
        else:
            refresh_interval = 30
            st.info("🔴 Auto-refresh disabled. Click button below to train manually.") 

        # Status indicators
        if st.session_state.get('model_trained', False):
            status_color = "status-active"
            status_text = "Active"
        else:
            status_color = "status-inactive"
            status_text = "Inactive"
        
        st.markdown(f"""
        <div class="refresh-toggle">
            <span class="status-indicator {status_color}"></span>
            <strong>Model Status:</strong> {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Safely display last update time
        if st.session_state.get('last_update') is not None:
            st.markdown(f"""
            <div class="small-text">
                Last update: {st.session_state.last_update.strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="small-text">
                Last update: Not trained yet
            </div>
            """, unsafe_allow_html=True)        
        
        st.info("""
        **Auto-Refresh Benefits:**
        - Adapts to market changes
        - Updates predictions real-time
        - Reduces manual intervention
        - Better for intraday trading
        
        **Note:** Auto-refresh helps capture sudden market movements
        """)
    
    # Check if we need to auto-refresh
    should_train = False
    
    if auto_refresh and st.session_state.get('model_trained', False):
        current_time = time.time()
        if st.session_state.get('last_train_time') is None:
            should_train = True
        elif current_time - st.session_state.last_train_time >= refresh_interval:
            should_train = True
    
    # Manual train button click
    if manual_train:
        should_train = True
    
    # Perform training if needed
    if should_train:
        # Reset the rerun counter when training happens to prevent duplicate messages
        st.session_state.rerun_counter = 0
        
        with st.spinner(f"Training AI model at {datetime.now().strftime('%H:%M:%S')}..."):
            # Initialize or reuse predictor
            if st.session_state.predictor is None or st.session_state.predictor.symbol != AutoRefreshStockPredictor.format_nse_symbol(symbol):
                st.session_state.predictor = AutoRefreshStockPredictor(symbol, interval, period)
            
            # Fetch latest data
            data = st.session_state.predictor.fetch_data()
            
            if data is not None and not data.empty:
                # Train model
                model = st.session_state.predictor.train_model()
                
                if model:
                    st.session_state.model_trained = True
                    st.session_state.predictions = st.session_state.predictor.predict()
                    st.session_state.last_update = datetime.now()
                    st.session_state.last_train_time = time.time()
                    
                    # Show success message (only on manual or first auto)
                    if manual_train:
                        st.success(f"✅ Model trained successfully at {st.session_state.last_update.strftime('%H:%M:%S')}")
                    elif not st.session_state.get('auto_refresh_shown', False):
                        st.success("Auto-refresh active - Model updating automatically")
                        st.session_state.auto_refresh_shown = True
                else:
                    st.error("Training failed - insufficient data (need at least 100 candles)")
            else:
                st.error(f"No data found for {symbol}")
    
    # Display current analysis
    if st.session_state.get('model_trained', False) and st.session_state.get('predictions') is not None:
        # Get current data for chart
        if st.session_state.predictor and st.session_state.predictor.data is not None:
            df_with_indicators = st.session_state.predictor.add_technical_indicators()
            
            if df_with_indicators is not None:
                # Show quick stats
                current_price = df_with_indicators['Close'].iloc[-1]
                price_change = df_with_indicators['Close'].iloc[-1] - df_with_indicators['Close'].iloc[-2] if len(df_with_indicators) > 1 else 0
                price_change_pct = (price_change / df_with_indicators['Close'].iloc[-2]) * 100 if len(df_with_indicators) > 1 else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change_pct:+.2f}%")
                with col2:
                    st.metric("Volume", f"{df_with_indicators['Volume'].iloc[-1]:,.0f}")
                with col3:
                    rsi_val = df_with_indicators['RSI'].iloc[-1] if 'RSI' in df_with_indicators.columns else 50
                    st.metric("RSI (14)", f"{rsi_val:.1f}")
                with col4:
                    atr_val = df_with_indicators['ATR'].iloc[-1] if 'ATR' in df_with_indicators.columns else current_price * 0.02
                    st.metric("ATR", f"₹{atr_val:.2f}")
                with col5:
                    stability = st.session_state.predictor.get_prediction_stability()
                    st.metric("Signal Stability", stability)
                
                # Display recommendations                
                st.markdown("## 🎯 LIVE TRADING RECOMMENDATIONS")
                display_recommendations(st.session_state.predictions)
                
                # Display chart                
                # st.markdown("## 📊 LIVE CHART")
                # fig = plot_chart(df_with_indicators, symbol)
                # st.plotly_chart(fig, width='stretch')
                
                # Auto-refresh countdown - Only show if auto_refresh is enabled
                if auto_refresh and st.session_state.get('model_trained', False):
                    time_since_update = time.time() - st.session_state.last_train_time
                    time_left = max(0, refresh_interval - time_since_update)
                    
                    st.markdown(f"""
                    <div class="update-time">
                        🔄 Next auto-refresh in {int(time_left)} seconds | 
                        Last update: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'Never'}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-refresh using rerun - only rerun once
                    if time_left <= 1 and st.session_state.rerun_counter == 0:
                        st.session_state.rerun_counter = 1
                        time.sleep(0.5)
                        st.rerun()
                    elif time_left > 1 and st.session_state.rerun_counter > 0:
                        # Reset rerun counter when we're not at the refresh point
                        st.session_state.rerun_counter = 0
                
                # Additional details in expander
                with st.expander("📊 Model Performance & History"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Prediction", st.session_state.predictions['prediction'])
                        st.metric("Confidence Level", f"{st.session_state.predictions['confidence']:.1f}%")
                        st.metric("Probability Score", f"{st.session_state.predictions['probability']:.1f}%")
                    
                    with col2:
                        st.metric("Raw AI Score", f"{st.session_state.predictions['raw_score']:.3f}")
                        st.metric("Support Level", f"₹{st.session_state.predictions['support']:.2f}")
                        st.metric("Resistance Level", f"₹{st.session_state.predictions['resistance']:.2f}")
                    
                    # Prediction history
                    if len(st.session_state.predictor.prediction_history) > 1:
                        st.markdown("#### Recent Prediction History")
                        history_df = pd.DataFrame([
                            {
                                'Time': p['timestamp'].strftime('%H:%M:%S'),
                                'Prediction': p['prediction'],
                                'Confidence': f"{p['confidence']:.0f}%",
                                'Price': f"₹{p['current_price']:.2f}"
                            }
                            for p in st.session_state.predictor.prediction_history[-10:]
                        ])
                        st.dataframe(history_df, width='stretch')
                
                # Feature importance if available
                if st.session_state.predictor.feature_importance:
                    with st.expander("🔬 Feature Importance Analysis"):
                        importance_df = pd.DataFrame(
                            list(st.session_state.predictor.feature_importance.items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False).head(8)
                        
                        fig_imp = go.Figure(data=[
                            go.Bar(x=importance_df['Importance'], 
                                  y=importance_df['Feature'],
                                  orientation='h',
                                  marker_color='#1e3c72')
                        ])
                        fig_imp.update_layout(
                            title="Top 8 Important Features",
                            template='plotly_white',
                            height=300,
                            font=dict(size=11)
                        )
                        st.plotly_chart(fig_imp, width='stretch')
    
    elif not st.session_state.get('model_trained', False):
        st.info("👈 Click 'Get Recommendation' or enable auto-refresh to start receiving AI predictions")
        
        # Example of what to expect
        st.markdown("""
        ### 🚀 Features with Auto-Refresh:
        
        - **Real-time Updates:** Model retrains every 30 seconds with latest data
        - **Live Predictions:** Get instant buy/sell signals based on current market
        - **Adaptive Learning:** Model adjusts to changing market conditions
        - **Risk Management:** Real-time stop loss and target calculations
        - **Signal Stability:** Monitor prediction consistency
        
        ### 💡 Best Practices:
        
        1. **Enable Auto-Refresh** for hands-free operation
        2. **Watch Signal Stability** - Wait for stable signals before trading
        3. **Use with 5m or 15m intervals** for best intraday results
        4. **Combine with technical analysis** from the chart
        5. **Always use stop losses** based on ATR calculations
        
        ### 📈 Recommended NSE Stocks:
        - RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
        - HINDUNILVR, SBIN, BHARTIARTL, KOTAKBANK, ITC
        """)

if __name__ == "__main__":
    main()
