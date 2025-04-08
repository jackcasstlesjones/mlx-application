import psycopg2
from psycopg2 import pool
from datetime import datetime
import os

# Database connection pool
db_pool = None

def init_db():
    """Initialize the database connection pool and create table if it doesn't exist"""
    global db_pool
    
    try:
        # Get database config from environment or use defaults
        host = os.environ.get("DB_HOST", "localhost")
        dbname = os.environ.get("DB_NAME", "mlx-db")
        user = os.environ.get("DB_USER", "postgres")  # Default to postgres user for container compatibility
        password = os.environ.get("DB_PASSWORD", "")
        port = os.environ.get("DB_PORT", "5432")
        
        # Create connection pool
        db_pool = pool.SimpleConnectionPool(
            1, 10,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    # Create predictions table if it doesn't exist
    with db_pool.getconn() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        predicted_digit INTEGER NOT NULL,
                        true_label INTEGER,
                        confidence REAL
                    );
                """)
            conn.commit()
        finally:
            db_pool.putconn(conn)
    
    print("Database initialized successfully")

def log_prediction(predicted_digit, true_label=None, confidence=None):
    """Log a prediction to the database
    
    Args:
        predicted_digit (int): The predicted digit
        true_label (int, optional): The user-provided true label
        confidence (float, optional): The confidence of the prediction
    """
    if db_pool is None:
        print("Database not initialized")
        return False
    
    try:
        with db_pool.getconn() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO predictions 
                        (timestamp, predicted_digit, true_label, confidence)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (datetime.now(), predicted_digit, true_label, confidence)
                    )
                conn.commit()
                print(f"Logged prediction: {predicted_digit}, true_label: {true_label}, confidence: {confidence}")
                return True
            except Exception as e:
                print(f"Error logging prediction: {e}")
                conn.rollback()
                return False
            finally:
                db_pool.putconn(conn)
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

def close_db():
    """Close the database connection pool"""
    if db_pool is not None:
        db_pool.closeall()
        print("Database connection closed")