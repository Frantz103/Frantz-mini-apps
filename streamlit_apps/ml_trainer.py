"""
ML Model Trainer
Train and evaluate simple ML models with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import plotly.express as px
import plotly.graph_objects as go


# Page config
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Model Trainer")
st.markdown("Train and evaluate machine learning models on your data.")
st.markdown("---")


# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    task_type = st.selectbox(
        "Select Task",
        ["Classification", "Regression"]
    )

    st.subheader("Data Settings")
    use_sample = st.checkbox("Use sample data", value=True)

    if use_sample:
        n_samples = st.slider("Number of samples", 100, 2000, 500)
        n_features = st.slider("Number of features", 2, 20, 10)

    st.subheader("Model Settings")

    if task_type == "Classification":
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression", "SVM"]
        )
    else:
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Linear Regression", "SVR"]
        )

    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", 0, 100, 42)


# Main content
def generate_sample_data(task, n_samples, n_features):
    """Generate sample dataset."""
    if task == "Classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features-5),
            n_redundant=min(2, n_features//4),
            random_state=random_state
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features-5),
            noise=10,
            random_state=random_state
        )

    # Create DataFrame
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y

    return df


def get_model(model_type, task):
    """Initialize model based on selection."""
    if task == "Classification":
        if model_type == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
        elif model_type == "Logistic Regression":
            return LogisticRegression(random_state=random_state, max_iter=1000)
        else:  # SVM
            return SVC(random_state=random_state)
    else:  # Regression
        if model_type == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10)
        elif model_type == "Linear Regression":
            return LinearRegression()
        else:  # SVR
            return SVR()


# Load or generate data
if use_sample:
    df = generate_sample_data(task_type, n_samples, n_features)
    st.success(f"✅ Generated sample dataset with {len(df)} samples and {len(df.columns)-1} features")
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded dataset with {len(df)} samples")
    else:
        st.info("👆 Please upload a CSV file")
        df = None


# Train model
if df is not None and st.button("🚀 Train Model", type="primary"):
    with st.spinner("Training model..."):
        # Prepare data
        X = df.drop('Target', axis=1)
        y = df['Target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        model = get_model(model_type, task_type)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Store results in session state
        st.session_state['model'] = model
        st.session_state['results'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
        }

    st.success("✅ Model trained successfully!")


# Display results
if 'results' in st.session_state:
    st.markdown("---")
    st.header("📊 Model Evaluation")

    results = st.session_state['results']

    # Metrics
    col1, col2 = st.columns(2)

    if task_type == "Classification":
        train_acc = accuracy_score(results['y_train'], results['y_pred_train'])
        test_acc = accuracy_score(results['y_test'], results['y_pred_test'])

        with col1:
            st.metric("Training Accuracy", f"{train_acc:.2%}")
        with col2:
            st.metric("Test Accuracy", f"{test_acc:.2%}")

        # Classification report
        st.subheader("Classification Report (Test Set)")
        report = classification_report(results['y_test'], results['y_pred_test'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    else:  # Regression
        train_rmse = np.sqrt(mean_squared_error(results['y_train'], results['y_pred_train']))
        test_rmse = np.sqrt(mean_squared_error(results['y_test'], results['y_pred_test']))
        train_r2 = r2_score(results['y_train'], results['y_pred_train'])
        test_r2 = r2_score(results['y_test'], results['y_pred_test'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training RMSE", f"{train_rmse:.2f}")
            st.metric("Training R²", f"{train_r2:.3f}")
        with col2:
            st.metric("Test RMSE", f"{test_rmse:.2f}")
            st.metric("Test R²", f"{test_r2:.3f}")

        # Predictions vs Actual
        st.subheader("Predictions vs Actual (Test Set)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['y_test'],
            y=results['y_pred_test'],
            mode='markers',
            name='Predictions'
        ))
        fig.add_trace(go.Scatter(
            x=results['y_test'],
            y=results['y_test'],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            title="Prediction Quality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (if available)
    if hasattr(st.session_state['model'], 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = st.session_state['model'].feature_importances_
        feature_names = results['X_train'].columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

elif df is not None:
    st.info("👆 Click 'Train Model' to start training!")
