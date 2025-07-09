# Loan Approval Prediction System
# Complete ML pipeline with web interface

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data preprocessing class
class LoanDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """Clean and preprocess the loan dataset"""
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in ['Loan_Status']:  # Don't encode target variable yet
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        return data
    
    def prepare_features(self, data, target_col='Loan_Status'):
        """Prepare features and target for training"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Encode target variable
        if target_col not in self.label_encoders:
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target_col] = le
        else:
            y = self.label_encoders[target_col].transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, X.columns.tolist()

# Model training class
class LoanPredictionModel:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.model_scores = {}
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train both models and evaluate performance"""
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            self.trained_models[name] = model
            self.model_scores[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
    def predict(self, model_name, X):
        """Make prediction using specified model"""
        if model_name in self.trained_models:
            return self.trained_models[model_name].predict(X)
        else:
            raise ValueError(f"Model {model_name} not found")

# Generate sample data (since we don't have the actual Kaggle dataset)
def generate_sample_data():
    """Generate sample loan data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.normal(5000, 2000, n_samples).astype(int),
        'CoapplicantIncome': np.random.normal(2000, 1500, n_samples).astype(int),
        'LoanAmount': np.random.normal(150, 50, n_samples).astype(int),
        'Loan_Amount_Term': np.random.choice([360, 240, 180, 120], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples)
    }
    
    # Create target variable with some logic
    df = pd.DataFrame(data)
    
    # Create loan status based on some realistic logic
    loan_status = []
    for i in range(n_samples):
        score = 0
        if df.iloc[i]['Credit_History'] == 1:
            score += 3
        if df.iloc[i]['Education'] == 'Graduate':
            score += 1
        if df.iloc[i]['ApplicantIncome'] > 4000:
            score += 1
        if df.iloc[i]['CoapplicantIncome'] > 1500:
            score += 1
        if df.iloc[i]['LoanAmount'] < 200:
            score += 1
        
        # Add some randomness
        if np.random.random() < 0.1:
            score = np.random.randint(0, 6)
        
        loan_status.append('Y' if score >= 3 else 'N')
    
    df['Loan_Status'] = loan_status
    return df

# Streamlit Web Interface
def create_streamlit_app():
    st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
    
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Model Training", "Prediction"])
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = generate_sample_data()
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = LoanDataPreprocessor()
    if 'model' not in st.session_state:
        st.session_state.model = LoanPredictionModel()
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    if page == "Data Overview":
        show_data_overview()
    elif page == "Model Training":
        show_model_training()
    elif page == "Prediction":
        show_prediction_page()

def show_data_overview():
    st.header("📊 Data Overview")
    
    data = st.session_state.data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Approved Loans", len(data[data['Loan_Status'] == 'Y']))
    with col3:
        st.metric("Rejected Loans", len(data[data['Loan_Status'] == 'N']))
    
    # Display data
    st.subheader("Dataset Sample")
    st.dataframe(data.head(10))
    
    # Data distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Loan status distribution
        fig = px.pie(data, names='Loan_Status', title='Loan Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Income distribution
        fig = px.histogram(data, x='ApplicantIncome', title='Applicant Income Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with loan approval
    st.subheader("Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit history vs approval
        ct = pd.crosstab(data['Credit_History'], data['Loan_Status'])
        fig = px.bar(ct, title='Credit History vs Loan Approval')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Education vs approval
        ct = pd.crosstab(data['Education'], data['Loan_Status'])
        fig = px.bar(ct, title='Education vs Loan Approval')
        st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    st.header("🤖 Model Training")
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            # Preprocess data
            preprocessor = st.session_state.preprocessor
            data = st.session_state.data
            
            processed_data = preprocessor.preprocess_data(data)
            X, y, feature_names = preprocessor.prepare_features(processed_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train models
            model = st.session_state.model
            model.train_models(X_train, X_test, y_train, y_test)
            
            st.session_state.trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_names
        
        st.success("Models trained successfully!")
    
    # Display results if trained
    if st.session_state.trained:
        st.subheader("Model Performance")
        
        model = st.session_state.model
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Logistic Regression Accuracy", 
                     f"{model.model_scores['Logistic Regression']['accuracy']:.4f}")
        
        with col2:
            st.metric("Random Forest Accuracy", 
                     f"{model.model_scores['Random Forest']['accuracy']:.4f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in model.trained_models:
            st.subheader("Feature Importance (Random Forest)")
            
            rf_model = model.trained_models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance, x='importance', y='feature', 
                        orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)

def show_prediction_page():
    st.header("🔮 Loan Prediction")
    
    if not st.session_state.trained:
        st.warning("Please train the models first!")
        return
    
    st.subheader("Enter Applicant Details")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=2000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
        loan_term = st.selectbox("Loan Amount Term", [360, 240, 180, 120])
        credit_history = st.selectbox("Credit History", [1, 0])
    
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
    
    if st.button("Predict Loan Approval"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })
        
        # Preprocess input
        preprocessor = st.session_state.preprocessor
        
        # Encode categorical variables
        for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            if col in preprocessor.label_encoders:
                input_data[col] = preprocessor.label_encoders[col].transform(input_data[col])
        
        # Scale features
        input_scaled = preprocessor.scaler.transform(input_data)
        
        # Make prediction
        model = st.session_state.model
        prediction = model.predict(model_choice, input_scaled)
        
        # Display result
        if prediction[0] == 1:
            st.success("✅ Loan Approved!")
            st.balloons()
        else:
            st.error("❌ Loan Rejected!")
        
        # Show prediction probability if available
        if hasattr(model.trained_models[model_choice], 'predict_proba'):
            proba = model.trained_models[model_choice].predict_proba(input_scaled)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rejection Probability", f"{proba[0]:.2%}")
            with col2:
                st.metric("Approval Probability", f"{proba[1]:.2%}")

if __name__ == "__main__":
    # For running the Streamlit app
    create_streamlit_app()
    
    # For standalone testing
    # Uncomment the following lines to run without Streamlit
    """
    # Generate sample data
    data = generate_sample_data()
    
    # Initialize preprocessor and model
    preprocessor = LoanDataPreprocessor()
    model = LoanPredictionModel()
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data(data)
    X, y, feature_names = preprocessor.prepare_features(processed_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    model.train_models(X_train, X_test, y_train, y_test)
    
    print("\\nTraining completed!")
    print("Use 'streamlit run loan_prediction.py' to launch the web interface")
    """