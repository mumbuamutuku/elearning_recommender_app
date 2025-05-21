"""
Database operations and initialization
"""

import sqlite3
from datetime import datetime

import pandas as pd

from config import COURSE_DATA
# import config

def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect('learning.db')
    # conn = sqlite3.connect(config.DB_NAME)
    return conn

def init_db():
    """Initialize the database schema"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS learners (
        id INTEGER PRIMARY KEY, 
        name TEXT, 
        age INTEGER, 
        goals TEXT, 
        preferences TEXT)''')
        
    c.execute('''CREATE TABLE IF NOT EXISTS courses (
        id TEXT PRIMARY KEY, 
        course_name TEXT, 
        course_description TEXT, 
        skills TEXT, 
        difficulty TEXT, 
        format TEXT, 
        course_rating REAL, 
        university TEXT, 
        course_url TEXT)''')
        
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        user_id INTEGER, 
        course_id INTEGER, 
        rating INTEGER, 
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        learner_id INTEGER, 
        course_id INTEGER, 
        rating INTEGER, 
        comment TEXT)''')
    
    # Ontology table (mock semantic web)
    c.execute('''CREATE TABLE IF NOT EXISTS prerequisites (
        course_id INTEGER, 
        prerequisite_id INTEGER)''')
        
    conn.commit()
    conn.close()

def populate_sample_data():
    """Fill database with sample data for demonstration"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Sample learners
    learners = [
        (1, 'Sarah', 25, 'Data Scientist', 'video'),
        (2, 'John', 30, 'Software Engineer', 'text'),
        (3, 'Emma', 22, 'Data Analyst', 'quiz')
    ]
    c.executemany('INSERT OR REPLACE INTO learners VALUES (?, ?, ?, ?, ?)', learners)
    
    # Sample courses
    # courses = [
    #     (1, 'Introduction to Python', 'python, programming', 'video'),
    #     (2, 'Data Visualization with Tableau', 'data science, visualization', 'video'),
    #     (3, 'Advanced Python', 'python, programming', 'text'),
    #     (4, 'Machine Learning Basics', 'machine learning, data science', 'quiz')
    # ]
    # c.executemany('INSERT OR REPLACE INTO courses VALUES (?, ?, ?, ?)', courses)
    try:
        df = pd.read_csv(COURSE_DATA)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '')
        courses = [(str(i + 1), row['course_name'], row['course_description'], row['skills'],
                    row.get('difficulty', row.get('difficulty_level', 'Unknown')),
                    'video' if 'video' in row.get('course_type', '').lower() else 'text',
                    row.get('course_rating', 0), row.get('university', 'Unknown'),
                    row.get('course_url', '#'))
                   for i, row in df.iterrows()]
    except FileNotFoundError:
        courses = [
            ('1', 'Introduction to Python', 'Learn Python programming basics', 'python, programming', 'Beginner', 'video', 4.0, 'University A', '#'),
            ('2', 'Data Visualization with Tableau', 'Visualize data effectively', 'data science, visualization', 'Intermediate', 'video', 4.5, 'University B', '#'),
            ('3', 'Advanced Python', 'Advanced Python concepts', 'python, programming', 'Intermediate', 'text', 4.2, 'University C', '#'),
            ('4', 'Machine Learning Basics', 'Introduction to ML', 'machine learning, data science', 'Intermediate', 'quiz', 4.0, 'University D', '#')
        ]
    # Sample interactions
    interactions = [
        (1, 1, 4, '2025-05-01 22:00:00'),  # Sarah rated Python course
        (1, 2, 5, '2025-05-01 22:30:00'),  # Sarah: Tableau
        (1, 3, 4, '2025-05-01 23:00:00'),
        (2, 3, 5, '2025-05-01 20:00:00'),  # John rated Advanced Python
        (2, 2, 4, '2025-05-01 21:00:00'),  # John: Tableau
        (2, 4, 4, '2025-05-01 21:30:00'),   # John: ML Basics
        (3, 4, 3, '2025-05-01 21:00:00'),   # Emma rated ML Basics
    ]
    c.executemany('INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?)', interactions)

    # Sample ontology
    prerequisites = [
        (2, 1),  # Data Visualization requires Python
        (3, 1),  # Advanced Python requires Python
        (4, 3),  # ML Basics requires Advanced Python
    ]
    c.executemany('INSERT OR REPLACE INTO prerequisites VALUES (?, ?)', prerequisites)
    
    conn.commit()
    conn.close()

def save_feedback(learner_id, course_id, rating, comment):
    """Save user feedback and update interactions"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Store the feedback
    c.execute('INSERT INTO feedback (learner_id, course_id, rating, comment) VALUES (?, ?, ?, ?)', 
              (learner_id, course_id, rating, comment))
              
    # Update the interactions table
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''INSERT OR REPLACE INTO interactions 
                 (learner_id, course_id, rating, time_spent, timestamp) 
                 VALUES (?, ?, ?, ?, ?)''', 
              (learner_id, course_id, rating, 10, timestamp))
    # Simulate federated learning: aggregate feedback locally
    c.execute('SELECT AVG(rating) FROM feedback WHERE course_id = ?', (course_id,))
    avg_rating = c.fetchone()[0] or rating
    c.execute('UPDATE interactions SET rating = ? WHERE course_id = ?', (avg_rating, course_id))
              
    conn.commit()
    conn.close()

def get_all_courses():
    """Retrieve all courses from the database"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM courses')
    courses = c.fetchall()
    conn.close()
    return courses

def get_all_interactions():
    """Retrieve all interaction data"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT user_id, course_id, rating FROM interactions')
    interactions = c.fetchall()
    conn.close()
    return interactions

def get_learner_feedback(learner_id):
    """Get feedback provided by a specific learner"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT course_id, rating FROM feedback WHERE learner_id = ?', (learner_id,))
    feedback = c.fetchall()
    conn.close()
    return feedback

def get_interaction_time(learner_id, course_ids):
    """Get the average time spent on specific courses by a learner"""
    if not course_ids:
        return 0
        
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create placeholders for SQL query
    placeholders = ','.join('?' * len(course_ids))
    
    c.execute(f'SELECT AVG(10) FROM interactions WHERE user_id = ? AND course_id IN ({placeholders})',
              [learner_id] + course_ids)
              
    result = c.fetchone()[0] or 0
    conn.close()
    return result