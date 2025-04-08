import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from db_utils import query_predictions, get_accuracy_stats, get_confusion_matrix

st.set_page_config(page_title="MNIST Predictions Analytics", layout="wide")
st.title("MNIST Prediction Analytics")

# Fetch prediction data
try:
    df = query_predictions(limit=1000)
    
    if df.empty:
        st.warning("No prediction data found in the database.")
    else:
        st.write(f"Loaded {len(df)} predictions from database")
        
        # Display recent predictions
        st.header("Recent Predictions")
        st.dataframe(df.head(10))
        
        # Accuracy statistics
        st.header("Accuracy Statistics")
        stats_df = get_accuracy_stats()
        
        if not stats_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", stats_df['total_predictions'].iloc[0])
            with col2:
                st.metric("Correct Predictions", stats_df['correct_predictions'].iloc[0])
            with col3:
                st.metric("Accuracy", f"{stats_df['accuracy_percentage'].iloc[0]}%")
        else:
            st.warning("No predictions with true labels found.")
        
        # Confusion Matrix
        st.header("Confusion Matrix")
        matrix = get_confusion_matrix()
        
        if not matrix.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
            plt.xlabel('Predicted Digit')
            plt.ylabel('True Digit')
            plt.title('Confusion Matrix')
            st.pyplot(fig)
        else:
            st.warning("Not enough data to generate a confusion matrix.")
        
        # Distribution of predictions
        st.header("Distribution of Predictions")
        fig, ax = plt.subplots(figsize=(10, 6))
        df['predicted_digit'].value_counts().sort_index().plot(kind='bar', ax=ax)
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Digits')
        st.pyplot(fig)
        
        # Confidence over time
        st.header("Prediction Confidence Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.scatter(df['timestamp'], df['confidence'], alpha=0.6)
        plt.xlabel('Time')
        plt.ylabel('Confidence')
        plt.title('Prediction Confidence Over Time')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    
st.sidebar.info("This dashboard visualizes predictions made by the MNIST digit recognition model.")