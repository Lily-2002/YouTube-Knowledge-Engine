import sqlite3
import os

def cleanup_database(db_path):
    # Verify the database file exists before connecting
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    try:
        # Establish connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL command to delete rows where response is 'ERROR'
        # Replace 'sentences' with your actual table name if it differs
        table_name = "sentences"
        sql_query = f"DELETE FROM {table_name} WHERE response NOT IN ('REMOVE', 'KEEP');"
        
        # Execute the deletion
        cursor.execute(sql_query)
        
        # Get the number of rows affected
        rows_deleted = cursor.rowcount
        
        # Commit the changes to the database
        conn.commit()
        
        print(f"Successfully deleted {rows_deleted} rows where response was not 'REMOVE' or 'KEEP'.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Always close the connection
        if conn:
            conn.close()

if __name__ == "__main__":
    path = "/home/fulian/RAG/data/request/sampled_sentences.db"
    cleanup_database(path)