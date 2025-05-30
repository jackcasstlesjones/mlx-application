import psycopg2
from psycopg2 import sql
import pandas as pd
import os

def get_connection_string():
    """Get the PostgreSQL connection string"""
    host = os.environ.get("DB_HOST", "localhost")
    dbname = os.environ.get("DB_NAME", "mlx-db")
    user = os.environ.get("DB_USER", "jack")  # Default to local user if not in container
    password = os.environ.get("DB_PASSWORD", "")
    port = os.environ.get("DB_PORT", "5432")
    
    # Build connection string with password only if it exists
    conn_string = f"dbname={dbname} user={user} host={host} port={port}"
    if password:
        conn_string += f" password={password}"
        
    return conn_string

def query_predictions(limit=100):
    """Query the most recent predictions"""
    try:
        with psycopg2.connect(get_connection_string()) as conn:
            query = """
            SELECT timestamp, predicted_digit, true_label, confidence 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT %s;
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df
    except Exception as e:
        print(f"Error querying predictions: {e}")
        return pd.DataFrame()

def get_accuracy_stats():
    """Get accuracy statistics for predictions with true labels"""
    try:
        with psycopg2.connect(get_connection_string()) as conn:
            query = """
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END) as correct_predictions,
                ROUND(SUM(CASE WHEN predicted_digit = true_label THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric * 100, 2) as accuracy_percentage
            FROM predictions
            WHERE true_label IS NOT NULL;
            """
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        print(f"Error getting accuracy stats: {e}")
        return pd.DataFrame()

def get_confusion_matrix():
    """Get a confusion matrix of predicted vs true labels"""
    try:
        with psycopg2.connect(get_connection_string()) as conn:
            query = """
            SELECT 
                predicted_digit,
                true_label,
                COUNT(*) as count
            FROM predictions
            WHERE true_label IS NOT NULL
            GROUP BY predicted_digit, true_label
            ORDER BY true_label, predicted_digit;
            """
            df = pd.read_sql_query(query, conn)
            # Convert to a pivot table format for easier visualization
            if not df.empty:
                matrix = df.pivot_table(
                    index='true_label', 
                    columns='predicted_digit', 
                    values='count', 
                    fill_value=0
                )
                return matrix
            return pd.DataFrame()
    except Exception as e:
        print(f"Error getting confusion matrix: {e}")
        return pd.DataFrame()