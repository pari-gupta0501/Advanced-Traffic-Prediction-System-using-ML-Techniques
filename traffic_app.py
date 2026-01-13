import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
from datetime import datetime
import json

st.set_page_config(page_title="🚦 Traffic Flow Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-message.assistant {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    .chat-message .message-header {
        font-weight: bold;
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚦 Advanced Traffic Flow Prediction System")
st.write("Predict traffic situations using deep learning with comprehensive analytics and AI chatbot assistance")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- Helper Functions --------------------
def create_sequences(data, sequence_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, num_classes, units=50):
    """Build LSTM model for classification"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_gru_model(input_shape, num_classes, units=50):
    """Build GRU model for classification"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Accuracy', 'Model Loss'))
    
    fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Accuracy',
                            mode='lines', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Val Accuracy',
                            mode='lines', line=dict(color='orange')), row=1, col=1)
    
    fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss',
                            mode='lines', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss',
                            mode='lines', line=dict(color='orange')), row=1, col=2)
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_layout(height=400, showlegend=True)
    
    return fig

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=classes,
                    y=classes,
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix", height=500)
    return fig

def plot_feature_distributions(df, features):
    """Plot feature distributions"""
    num_features = len(features)
    cols = min(3, num_features)
    rows = (num_features + cols - 1) // cols
    
    fig = make_subplots(rows=rows, cols=cols, 
                        subplot_titles=features,
                        vertical_spacing=0.1)
    
    for idx, feature in enumerate(features):
        row = idx // cols + 1
        col = idx % cols + 1
        
        if df[feature].dtype in ['int64', 'float64']:
            fig.add_trace(go.Histogram(x=df[feature], name=feature, 
                                      showlegend=False), row=row, col=col)
        else:
            value_counts = df[feature].value_counts()
            fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values,
                               name=feature, showlegend=False), row=row, col=col)
    
    fig.update_layout(height=300*rows, showlegend=False, title_text="Feature Distributions")
    return fig

def generate_chatbot_response(user_message, df, context):
    """Generate intelligent responses based on user queries"""
    user_message_lower = user_message.lower()
    
    # Dataset information queries
    if any(word in user_message_lower for word in ['dataset', 'data', 'rows', 'columns', 'features']):
        if df is not None:
            response = f"📊 **Dataset Information:**\n\n"
            response += f"- Total records: {len(df):,}\n"
            response += f"- Total features: {len(df.columns)}\n"
            response += f"- Numerical features: {len(df.select_dtypes(include=[np.number]).columns)}\n"
            response += f"- Categorical features: {len(df.select_dtypes(include=['object']).columns)}\n"
            response += f"- Column names: {', '.join(df.columns.tolist())}\n\n"
            
            missing = df.isnull().sum().sum()
            if missing > 0:
                response += f"- Missing values: {missing} ({(missing/(len(df)*len(df.columns))*100):.2f}%)\n"
            else:
                response += "- No missing values found! ✅\n"
            return response
        else:
            return "⚠️ No dataset is currently loaded. Please upload a CSV file first."
    
    # Model status queries
    elif any(word in user_message_lower for word in ['model', 'trained', 'accuracy', 'performance']):
        if "model" in st.session_state:
            response = f"🤖 **Model Information:**\n\n"
            response += f"- Model Type: {st.session_state['model_type']}\n"
            response += f"- Accuracy: {st.session_state['accuracy']:.2%}\n"
            response += f"- Features used: {len(st.session_state['features'])} ({', '.join(st.session_state['features'][:3])}...)\n"
            response += f"- Target variable: {st.session_state['target']}\n"
            response += f"- Trained at: {st.session_state['trained_at']}\n\n"
            response += "The model is ready to make predictions! Go to the 🔮 Predictions page."
            return response
        else:
            return "⚠️ No model has been trained yet. Go to the 🤖 Model Training page to train a model first."
    
    # Prediction help
    elif any(word in user_message_lower for word in ['predict', 'prediction', 'forecast']):
        if "model" in st.session_state:
            response = "🔮 **Making Predictions:**\n\n"
            response += "1. Navigate to the **🔮 Predictions** page\n"
            response += "2. Choose between Single or Batch prediction\n"
            if st.session_state['model_type'] in ["LSTM", "GRU"]:
                response += f"3. For sequential models, enter {st.session_state['sequence_length']} time steps\n"
            else:
                response += "3. Enter values for each feature\n"
            response += "4. Click the Predict button\n\n"
            response += "The system will show the predicted class with confidence scores!"
            return response
        else:
            return "⚠️ Please train a model first before making predictions."
    
    # Model selection advice
    elif any(word in user_message_lower for word in ['which model', 'best model', 'choose model', 'lstm', 'gru', 'random forest']):
        response = "🧠 **Model Selection Guide:**\n\n"
        response += "**LSTM (Long Short-Term Memory):**\n"
        response += "- Best for: Sequential/time-series data with long-term dependencies\n"
        response += "- Pros: Captures temporal patterns, handles sequences well\n"
        response += "- Training time: Slower\n\n"
        response += "**GRU (Gated Recurrent Unit):**\n"
        response += "- Best for: Time-series data, faster than LSTM\n"
        response += "- Pros: Simpler architecture, faster training\n"
        response += "- Training time: Medium\n\n"
        response += "**Random Forest:**\n"
        response += "- Best for: Non-sequential data, feature importance analysis\n"
        response += "- Pros: Fast, interpretable, no sequence needed\n"
        response += "- Training time: Fastest\n\n"
        response += "💡 For traffic data with time dependencies, try LSTM or GRU first!"
        return response
    
    # Hyperparameter tuning
    elif any(word in user_message_lower for word in ['hyperparameter', 'epochs', 'batch size', 'units', 'parameters']):
        response = "⚙️ **Hyperparameter Tuning Tips:**\n\n"
        response += "**Epochs:** Number of training iterations\n"
        response += "- Start with 50-100 for neural networks\n"
        response += "- More epochs = better learning (but risk overfitting)\n\n"
        response += "**Batch Size:** Number of samples per training step\n"
        response += "- Typical values: 32, 64, 128\n"
        response += "- Smaller = more accurate but slower\n\n"
        response += "**Network Units:** Number of neurons in layers\n"
        response += "- Start with 50-100 units\n"
        response += "- More units = more capacity (but slower)\n\n"
        response += "**Sequence Length:** For LSTM/GRU only\n"
        response += "- How many past time steps to consider\n"
        response += "- Typical range: 10-30\n\n"
        response += "💡 Start with defaults and adjust based on performance!"
        return response
    
    # Feature engineering
    elif any(word in user_message_lower for word in ['feature', 'feature engineering', 'improve accuracy']):
        response = "🔧 **Feature Engineering Tips:**\n\n"
        response += "1. **Feature Selection:** Choose relevant features that impact traffic\n"
        response += "   - Time-based: Hour, Day, Month\n"
        response += "   - Traffic metrics: Speed, Volume, Density\n"
        response += "   - Environmental: Weather, Events\n\n"
        response += "2. **Feature Scaling:** Already handled by the system ✅\n\n"
        response += "3. **Handle Missing Data:**\n"
        response += "   - Check Data Explorer for missing values\n"
        response += "   - Consider removing or imputing\n\n"
        response += "4. **Create New Features:**\n"
        response += "   - Rush hour indicators\n"
        response += "   - Weekend flags\n"
        response += "   - Rolling averages\n\n"
        response += "💡 More relevant features = better predictions!"
        return response
    
    # How to use the app
    elif any(word in user_message_lower for word in ['how to', 'how do i', 'help', 'guide', 'tutorial', 'start']):
        response = "📚 **Quick Start Guide:**\n\n"
        response += "**Step 1: Load Data** 📊\n"
        response += "- Upload CSV or use default dataset\n"
        response += "- Explore in Data Explorer page\n\n"
        response += "**Step 2: Train Model** 🤖\n"
        response += "- Go to Model Training page\n"
        response += "- Select target and features\n"
        response += "- Choose model type (LSTM/GRU/Random Forest)\n"
        response += "- Adjust hyperparameters\n"
        response += "- Click Train Model button\n\n"
        response += "**Step 3: Make Predictions** 🔮\n"
        response += "- Navigate to Predictions page\n"
        response += "- Enter input values\n"
        response += "- Get predictions with confidence scores\n\n"
        response += "**Step 4: Analyze Results** 📈\n"
        response += "- Check Analytics page for insights\n"
        response += "- View confusion matrix and metrics\n\n"
        response += "**Step 5: Save/Load Models** 💾\n"
        response += "- Save trained models for later use\n"
        response += "- Load previously saved models\n\n"
        response += "Need help with something specific? Just ask! 😊"
        return response
    
    # Accuracy improvement
    elif any(word in user_message_lower for word in ['low accuracy', 'poor performance', 'improve', 'better results']):
        response = "📈 **Tips to Improve Model Accuracy:**\n\n"
        response += "1. **More Training Data:** Add more samples if possible\n\n"
        response += "2. **Feature Selection:** Try different feature combinations\n\n"
        response += "3. **Hyperparameter Tuning:**\n"
        response += "   - Increase epochs (100-200)\n"
        response += "   - Try different batch sizes\n"
        response += "   - Adjust network units (75-150)\n\n"
        response += "4. **Try Different Models:**\n"
        response += "   - Compare LSTM, GRU, and Random Forest\n"
        response += "   - Each works better for different data patterns\n\n"
        response += "5. **Data Quality:**\n"
        response += "   - Remove outliers\n"
        response += "   - Handle missing values\n"
        response += "   - Balance class distribution\n\n"
        response += "6. **Sequence Length:** For LSTM/GRU, try different lengths\n\n"
        response += "💡 Experiment with these options systematically!"
        return response
    
    # Error handling help
    elif any(word in user_message_lower for word in ['error', 'problem', 'not working', 'issue']):
        response = "🔧 **Troubleshooting Common Issues:**\n\n"
        response += "**Training Errors:**\n"
        response += "- Ensure all features have valid data\n"
        response += "- Check for missing values in dataset\n"
        response += "- Verify target column has categorical values\n\n"
        response += "**Prediction Errors:**\n"
        response += "- Train a model first before predicting\n"
        response += "- Use same features as during training\n"
        response += "- Enter valid values for all inputs\n\n"
        response += "**Performance Issues:**\n"
        response += "- Reduce epochs for faster training\n"
        response += "- Use smaller batch sizes\n"
        response += "- Simplify feature selection\n\n"
        response += "Still having issues? Check the error details in the expander below error messages."
        return response
    
    # Traffic-specific queries
    elif any(word in user_message_lower for word in ['traffic', 'congestion', 'flow', 'jam']):
        response = "🚦 **Traffic Flow Prediction:**\n\n"
        response += "This system helps predict traffic situations based on historical patterns.\n\n"
        response += "**Common Traffic Classes:**\n"
        response += "- Free flow: Light traffic, high speeds\n"
        response += "- Moderate: Normal traffic conditions\n"
        response += "- Heavy: Congested, slower speeds\n"
        response += "- Jammed: Severe congestion\n\n"
        response += "**Key Factors:**\n"
        response += "- Time of day (rush hours)\n"
        response += "- Day of week (weekday vs weekend)\n"
        response += "- Weather conditions\n"
        response += "- Special events\n"
        response += "- Historical patterns\n\n"
        response += "💡 Train your model with diverse traffic scenarios for best results!"
        return response
    
    # Greeting
    elif any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "👋 Hello! I'm your Traffic Prediction AI Assistant. I can help you with:\n\n- Understanding the dataset\n- Training models\n- Making predictions\n- Improving accuracy\n- Troubleshooting issues\n\nWhat would you like to know?"
    
    # Thanks
    elif any(word in user_message_lower for word in ['thank', 'thanks', 'appreciate']):
        return "😊 You're welcome! Feel free to ask if you have any other questions about traffic prediction or the system!"
    
    # Default response
    else:
        return "🤔 I'm here to help with traffic prediction! You can ask me about:\n\n" \
               "- Dataset information and statistics\n" \
               "- Model training and selection\n" \
               "- Making predictions\n" \
               "- Improving model accuracy\n" \
               "- Hyperparameter tuning\n" \
               "- Feature engineering\n" \
               "- Troubleshooting errors\n\n" \
               "What would you like to know?"

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("TrafficDataset.csv")
        return df
    except FileNotFoundError:
        return None

# Sidebar for navigation
with st.sidebar:
    st.header("🎛️ Navigation")
    page = st.radio("Select Page", 
                    ["💬 AI Chatbot", "📊 Data Explorer", "🤖 Model Training", 
                     "🔮 Predictions", "📈 Analytics", "💾 Model Management"])
    
    st.divider()
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is None and page != "💬 AI Chatbot":
    st.warning("⚠️ No dataset loaded. Please upload a CSV file to begin.")
    st.stop()

# ==================== AI CHATBOT ====================
if page == "💬 AI Chatbot":
    st.header("💬 AI Traffic Prediction Assistant")
    st.write("Ask me anything about traffic prediction, models, or how to use this system!")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-header">👤 You</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="message-header">🤖 AI Assistant</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Type your question here...", key="chat_input", label_visibility="collapsed")
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Quick action buttons
    st.write("**Quick Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Dataset Info"):
            user_input = "Tell me about the dataset"
            send_button = True
    with col2:
        if st.button("🤖 Model Status"):
            user_input = "What's the model status?"
            send_button = True
    with col3:
        if st.button("📚 How to Start"):
            user_input = "How do I start?"
            send_button = True
    with col4:
        if st.button("💡 Improve Accuracy"):
            user_input = "How to improve accuracy?"
            send_button = True
    
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        context = {
            "has_model": "model" in st.session_state,
            "has_data": df is not None
        }
        
        response = generate_chatbot_response(user_input, df, context)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun to update chat display
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - What information is in my dataset?
        - How do I train a model?
        - Which model should I choose?
        - What are the best hyperparameters?
        - How can I improve model accuracy?
        - How do I make predictions?
        - What is the difference between LSTM and GRU?
        - How to handle missing values?
        - Tell me about feature engineering
        """)

# ==================== DATA EXPLORER ====================
elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Statistics", "Visualizations"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Numerical Features", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({'Feature': missing.index, 'Missing Count': missing.values, 
                                  'Percentage': (missing.values / len(df) * 100).round(2)})
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Categorical Features Summary")
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            for col in cat_cols:
                with st.expander(f"📊 {col}"):
                    value_counts = df[col].value_counts()
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   labels={'x': col, 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(value_counts.reset_index(name='count'), use_container_width=True)
    
    with tab3:
        st.subheader("Feature Distributions")
        selected_features = st.multiselect("Select features to visualize", 
                                          df.columns.tolist(),
                                          default=df.columns.tolist()[:6])
        if selected_features:
            fig = plot_feature_distributions(df, selected_features)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

# ==================== MODEL TRAINING ====================
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    all_columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("🎯 Target Column", all_columns, 
                             index=len(all_columns)-1)
    with col2:
        model_type = st.selectbox("🧠 Model Type", ["LSTM", "GRU", "Random Forest"])
    
    available_features = [c for c in all_columns if c != target]
    features = st.multiselect("📋 Select Features", available_features, 
                             default=available_features[:min(5, len(available_features))])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sequence_length = st.slider("⏱️ Sequence Length", 5, 30, 10)
    with col2:
        epochs = st.slider("🔄 Training Epochs", 10, 200, 50, step=10)
    with col3:
        batch_size = st.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
    
    if model_type in ["LSTM", "GRU"]:
        units = st.slider("🧮 Network Units", 25, 200, 50, step=25)
    
    if features and target:
        if df[target].dtype == 'object':
            unique_classes = list(df[target].unique())
            st.info(f"🎯 **Classification Task**: Predicting {len(unique_classes)} classes: {', '.join(map(str, unique_classes))}")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Prepare data
                    status_text.text("Preparing data...")
                    progress_bar.progress(10)
                    
                    X = df[features].copy()
                    y = df[target].copy()
                    
                    # Handle categorical features
                    X_processed = X.copy()
                    feature_encoders = {}
                    
                    for col in X_processed.columns:
                        if X_processed[col].dtype == 'object':
                            encoder = LabelEncoder()
                            X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
                            feature_encoders[col] = encoder
                    
                    # Scale features
                    feature_scaler = MinMaxScaler()
                    X_scaled = feature_scaler.fit_transform(X_processed)
                    
                    progress_bar.progress(20)
                    
                    # Handle target
                    if y.dtype == 'object':
                        target_encoder = LabelEncoder()
                        y_encoded = target_encoder.fit_transform(y.astype(str))
                        num_classes = len(target_encoder.classes_)
                        
                        if model_type in ["LSTM", "GRU"]:
                            y_onehot = to_categorical(y_encoded)
                            
                            status_text.text("Creating sequences...")
                            progress_bar.progress(30)
                            
                            X_seq, _ = create_sequences(X_scaled, sequence_length)
                            y_seq = y_onehot[sequence_length:]
                            
                            if len(X_seq) != len(y_seq):
                                min_len = min(len(X_seq), len(y_seq))
                                X_seq = X_seq[:min_len]
                                y_seq = y_seq[:min_len]
                            
                            train_size = int(len(X_seq) * 0.8)
                            X_train = X_seq[:train_size]
                            X_test = X_seq[train_size:]
                            y_train = y_seq[:train_size]
                            y_test = y_seq[train_size:]
                            
                            status_text.text(f"Building {model_type} model...")
                            progress_bar.progress(40)
                            
                            if model_type == "LSTM":
                                model = build_lstm_model((sequence_length, X_scaled.shape[1]), num_classes, units)
                            else:  # GRU
                                model = build_gru_model((sequence_length, X_scaled.shape[1]), num_classes, units)
                            
                            status_text.text("Training neural network...")
                            progress_bar.progress(50)
                            
                            history = model.fit(X_train, y_train, 
                                              epochs=epochs, 
                                              batch_size=batch_size, 
                                              validation_split=0.2, 
                                              verbose=0)
                            
                            progress_bar.progress(90)
                            
                            y_pred = model.predict(X_test, verbose=0)
                            y_pred_classes = np.argmax(y_pred, axis=1)
                            y_test_classes = np.argmax(y_test, axis=1)
                            
                            st.session_state["model"] = model
                            st.session_state["history"] = history
                            st.session_state["sequence_length"] = sequence_length
                            
                        else:  # Random Forest
                            status_text.text("Training Random Forest...")
                            progress_bar.progress(50)
                            
                            train_size = int(len(X_scaled) * 0.8)
                            X_train = X_scaled[:train_size]
                            X_test = X_scaled[train_size:]
                            y_train = y_encoded[:train_size]
                            y_test = y_encoded[train_size:]
                            
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
                            progress_bar.progress(90)
                            
                            y_pred_classes = model.predict(X_test)
                            y_test_classes = y_test
                            
                            st.session_state["model"] = model
                            st.session_state["history"] = None
                        
                        accuracy = accuracy_score(y_test_classes, y_pred_classes)
                        
                        progress_bar.progress(100)
                        status_text.text("Training complete!")
                        
                        st.success(f"✅ Model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Test Samples", len(y_test_classes))
                        with col3:
                            st.metric("Classes", num_classes)
                        
                        # Classification report
                        report = classification_report(y_test_classes, y_pred_classes, 
                                                     target_names=target_encoder.classes_, 
                                                     output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                        
                        # Confusion matrix
                        st.subheader("Confusion Matrix")
                        fig_cm = plot_confusion_matrix(y_test_classes, y_pred_classes, target_encoder.classes_)
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Training history for neural networks
                        if model_type in ["LSTM", "GRU"] and "history" in st.session_state:
                            st.subheader("Training History")
                            fig_history = plot_training_history(st.session_state["history"])
                            st.plotly_chart(fig_history, use_container_width=True)
                        
                        # Store session state
                        st.session_state["feature_scaler"] = feature_scaler
                        st.session_state["feature_encoders"] = feature_encoders
                        st.session_state["target_encoder"] = target_encoder
                        st.session_state["features"] = features
                        st.session_state["target"] = target
                        st.session_state["model_type"] = model_type
                        st.session_state["accuracy"] = accuracy
                        st.session_state["y_test"] = y_test_classes
                        st.session_state["y_pred"] = y_pred_classes
                        st.session_state["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

# ==================== PREDICTIONS ====================
elif page == "🔮 Predictions":
    st.header("🔮 Make Predictions")
    
    if "model" not in st.session_state:
        st.warning("⚠️ Please train a model first from the Model Training page.")
        st.stop()
    
    st.info(f"📊 Using {st.session_state['model_type']} model trained on {st.session_state['trained_at']}")
    st.write(f"🎯 Predicting: **{st.session_state['target']}** | Accuracy: **{st.session_state['accuracy']:.2%}**")
    
    prediction_mode = st.radio("Select Prediction Mode", 
                               ["Single Prediction", "Batch Prediction"])
    
    if prediction_mode == "Single Prediction":
        if st.session_state["model_type"] in ["LSTM", "GRU"]:
            st.write(f"Enter {st.session_state['sequence_length']} sequential time steps:")
            
            sequence_inputs = []
            for i in range(st.session_state['sequence_length']):
                with st.expander(f"⏱️ Time Step {i+1}", expanded=(i==0)):
                    step_inputs = {}
                    cols = st.columns(len(st.session_state["features"]))
                    
                    for j, feature in enumerate(st.session_state["features"]):
                        if feature in st.session_state["feature_encoders"]:
                            options = list(df[feature].unique())
                            value = cols[j].selectbox(f"{feature}", options, key=f"{feature}_step_{i}")
                            step_inputs[feature] = value
                        else:
                            default_val = float(df[feature].median())
                            value = cols[j].number_input(f"{feature}", value=default_val, key=f"{feature}_step_{i}")
                            step_inputs[feature] = value
                    
                    sequence_inputs.append(step_inputs)
            
            if st.button("🔮 Predict", type="primary"):
                try:
                    sequence_data = []
                    for step in sequence_inputs:
                        step_values = []
                        for feature in st.session_state["features"]:
                            if feature in st.session_state["feature_encoders"]:
                                encoded_val = st.session_state["feature_encoders"][feature].transform([str(step[feature])])[0]
                                step_values.append(encoded_val)
                            else:
                                step_values.append(float(step[feature]))
                        sequence_data.append(step_values)
                    
                    sequence_array = np.array(sequence_data, dtype=np.float32)
                    sequence_scaled = st.session_state["feature_scaler"].transform(sequence_array)
                    sequence_reshaped = sequence_scaled.reshape(1, st.session_state['sequence_length'], -1)
                    
                    prediction = st.session_state["model"].predict(sequence_reshaped, verbose=0)
                    predicted_class_idx = np.argmax(prediction, axis=1)[0]
                    predicted_class = st.session_state["target_encoder"].classes_[predicted_class_idx]
                    confidence = prediction[0][predicted_class_idx] * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"### 🚦 Predicted: **{predicted_class}**")
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col2:
                        st.write("### 📊 All Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': st.session_state["target_encoder"].classes_,
                            'Probability': prediction[0] * 100
                        }).sort_values('Probability', ascending=False)
                        
                        fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                                   color='Probability', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        else:  # Random Forest
            st.write("Enter feature values:")
            input_data = {}
            cols = st.columns(len(st.session_state["features"]))
            
            for j, feature in enumerate(st.session_state["features"]):
                if feature in st.session_state["feature_encoders"]:
                    options = list(df[feature].unique())
                    value = cols[j].selectbox(f"{feature}", options)
                    input_data[feature] = value
                else:
                    default_val = float(df[feature].median())
                    value = cols[j].number_input(f"{feature}", value=default_val)
                    input_data[feature] = value
            
            if st.button("🔮 Predict", type="primary"):
                try:
                    input_values = []
                    for feature in st.session_state["features"]:
                        if feature in st.session_state["feature_encoders"]:
                            encoded_val = st.session_state["feature_encoders"][feature].transform([str(input_data[feature])])[0]
                            input_values.append(encoded_val)
                        else:
                            input_values.append(float(input_data[feature]))
                    
                    input_array = np.array([input_values], dtype=np.float32)
                    input_scaled = st.session_state["feature_scaler"].transform(input_array)
                    
                    prediction = st.session_state["model"].predict(input_scaled)
                    predicted_class = st.session_state["target_encoder"].classes_[prediction[0]]
                    
                    proba = st.session_state["model"].predict_proba(input_scaled)[0]
                    confidence = proba[prediction[0]] * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"### 🚦 Predicted: **{predicted_class}**")
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col2:
                        st.write("### 📊 All Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': st.session_state["target_encoder"].classes_,
                            'Probability': proba * 100
                        }).sort_values('Probability', ascending=False)
                        
                        fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                                   color='Probability', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    else:  # Batch Prediction
        st.subheader("📦 Batch Prediction")
        batch_file = st.file_uploader("Upload CSV file for batch predictions", type=['csv'])
        
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Predictions"):
                st.info("Batch prediction feature coming soon!")

# ==================== ANALYTICS ====================
elif page == "📈 Analytics":
    st.header("📈 Model Analytics")
    
    if "model" not in st.session_state:
        st.warning("⚠️ Please train a model first from the Model Training page.")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Feature Analysis", "Prediction Analysis"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Type", st.session_state["model_type"])
        with col2:
            st.metric("Accuracy", f"{st.session_state['accuracy']:.2%}")
        with col3:
            st.metric("Features Used", len(st.session_state["features"]))
        with col4:
            st.metric("Target", st.session_state["target"])
        
        if "history" in st.session_state and st.session_state["history"]:
            st.subheader("Training History")
            fig = plot_training_history(st.session_state["history"])
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(st.session_state["y_test"], 
                                      st.session_state["y_pred"], 
                                      st.session_state["target_encoder"].classes_)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance")
        
        if st.session_state["model_type"] == "Random Forest":
            importance = st.session_state["model"].feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': st.session_state["features"],
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance analysis is available for Random Forest models.")
        
        st.subheader("Feature Distributions in Dataset")
        fig = plot_feature_distributions(df, st.session_state["features"])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Prediction Distribution")
        
        pred_counts = pd.Series(st.session_state["y_pred"]).value_counts()
        pred_labels = [st.session_state["target_encoder"].classes_[i] for i in pred_counts.index]
        
        fig = px.pie(values=pred_counts.values, names=pred_labels, 
                    title="Distribution of Predictions on Test Set")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Actual vs Predicted")
        comparison_df = pd.DataFrame({
            'Actual': [st.session_state["target_encoder"].classes_[i] for i in st.session_state["y_test"][:50]],
            'Predicted': [st.session_state["target_encoder"].classes_[i] for i in st.session_state["y_pred"][:50]]
        })
        comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Predicted']
        comparison_df.index.name = 'Sample'
        
        st.dataframe(comparison_df.style.applymap(
            lambda x: 'background-color: lightgreen' if x == True else ('background-color: lightcoral' if x == False else ''),
            subset=['Match']
        ), use_container_width=True)

# ==================== MODEL MANAGEMENT ====================
elif page == "💾 Model Management":
    st.header("💾 Model Management")
    
    tab1, tab2 = st.tabs(["Save Model", "Load Model"])
    
    with tab1:
        st.subheader("Save Trained Model")
        
        if "model" not in st.session_state:
            st.warning("⚠️ No trained model found. Please train a model first.")
        else:
            st.success("✅ Model ready to save")
            
            model_name = st.text_input("Model Name", value=f"traffic_model_{st.session_state['model_type'].lower()}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Information:**")
                st.write(f"- Type: {st.session_state['model_type']}")
                st.write(f"- Accuracy: {st.session_state['accuracy']:.2%}")
                st.write(f"- Features: {len(st.session_state['features'])}")
                st.write(f"- Target: {st.session_state['target']}")
                st.write(f"- Trained: {st.session_state['trained_at']}")
            
            with col2:
                st.write("**What will be saved:**")
                st.write("✓ Trained model")
                st.write("✓ Feature scalers")
                st.write("✓ Label encoders")
                st.write("✓ Feature list")
                st.write("✓ Model configuration")
            
            if st.button("💾 Save Model Package", type="primary"):
                try:
                    # Create model package
                    model_package = {
                        'model_type': st.session_state['model_type'],
                        'features': st.session_state['features'],
                        'target': st.session_state['target'],
                        'feature_scaler': st.session_state['feature_scaler'],
                        'feature_encoders': st.session_state['feature_encoders'],
                        'target_encoder': st.session_state['target_encoder'],
                        'accuracy': st.session_state['accuracy'],
                        'trained_at': st.session_state['trained_at']
                    }
                    
                    if st.session_state['model_type'] in ["LSTM", "GRU"]:
                        model_package['sequence_length'] = st.session_state['sequence_length']
                    
                    # Serialize model package
                    package_bytes = pickle.dumps(model_package)
                    
                    # Create download button for package
                    st.download_button(
                        label="📥 Download Model Package (.pkl)",
                        data=package_bytes,
                        file_name=f"{model_name}_package.pkl",
                        mime="application/octet-stream"
                    )
                    
                    # Save neural network model separately if applicable
                    if st.session_state['model_type'] in ["LSTM", "GRU"]:
                        # Save model architecture and weights
                        model_buffer = io.BytesIO()
                        st.session_state['model'].save(model_buffer)
                        model_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Neural Network Model (.h5)",
                            data=model_buffer,
                            file_name=f"{model_name}_model.h5",
                            mime="application/octet-stream"
                        )
                    
                    st.success("✅ Model package prepared for download!")
                    st.info("💡 Download both files to fully restore the model later.")
                    
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
    
    with tab2:
        st.subheader("Load Saved Model")
        
        st.info("📤 Upload the model package (.pkl) file to restore a trained model")
        
        uploaded_package = st.file_uploader("Upload Model Package", type=['pkl'])
        uploaded_model = st.file_uploader("Upload Neural Network Model (if applicable)", type=['h5'])
        
        if uploaded_package:
            if st.button("📂 Load Model", type="primary"):
                try:
                    # Load model package
                    model_package = pickle.loads(uploaded_package.read())
                    
                    # Restore session state
                    st.session_state['model_type'] = model_package['model_type']
                    st.session_state['features'] = model_package['features']
                    st.session_state['target'] = model_package['target']
                    st.session_state['feature_scaler'] = model_package['feature_scaler']
                    st.session_state['feature_encoders'] = model_package['feature_encoders']
                    st.session_state['target_encoder'] = model_package['target_encoder']
                    st.session_state['accuracy'] = model_package['accuracy']
                    st.session_state['trained_at'] = model_package['trained_at']
                    
                    if 'sequence_length' in model_package:
                        st.session_state['sequence_length'] = model_package['sequence_length']
                    
                    # Load neural network model if provided
                    if uploaded_model and model_package['model_type'] in ["LSTM", "GRU"]:
                        model = tf.keras.models.load_model(uploaded_model)
                        st.session_state['model'] = model
                    elif model_package['model_type'] == "Random Forest":
                        # For Random Forest, it's stored in the package
                        st.session_state['model'] = model_package.get('model')
                    
                    st.success("✅ Model loaded successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Loaded Model Information:**")
                        st.write(f"- Type: {st.session_state['model_type']}")
                        st.write(f"- Accuracy: {st.session_state['accuracy']:.2%}")
                        st.write(f"- Features: {len(st.session_state['features'])}")
                    with col2:
                        st.write(f"- Target: {st.session_state['target']}")
                        st.write(f"- Originally Trained: {st.session_state['trained_at']}")
                        st.write("- Status: Ready for predictions")
                    
                    st.info("✨ You can now use the Predictions page to make predictions with this model!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    with st.expander("View Error Details"):
                        import traceback
                        st.code(traceback.format_exc())

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🚦 Advanced Traffic Flow Prediction System with AI Chatbot | Built with Streamlit & TensorFlow</p>
    <p>💡 <b>Tips:</b> Use the AI Chatbot for instant help | Try different model types and compare performance</p>
</div>
""", unsafe_allow_html=True)
