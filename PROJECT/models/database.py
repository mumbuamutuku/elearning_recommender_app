import sqlite3
import pandas as pd
from config import DATABASE, COURSE_DATA

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS learners (
        id INTEGER PRIMARY KEY, name TEXT, age INTEGER, goals TEXT, preferences TEXT, study_hours INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY, title TEXT, description TEXT, skills TEXT, difficulty TEXT, format TEXT, rating REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        learner_id INTEGER, course_id INTEGER, rating INTEGER, time_spent INTEGER, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        learner_id INTEGER, course_id INTEGER, rating INTEGER, comment TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ontology (
        course_id INTEGER, prerequisite_id INTEGER)''')
    conn.commit()
    conn.close()

def populate_sample_data():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # Learners
    learners = [(1, 'Sarah', 25, 'Data Scientist', 'video', 2)]
    c.executemany('INSERT OR REPLACE INTO learners VALUES (?, ?, ?, ?, ?, ?)', learners)
    # Courses
    try:
        df = pd.read_csv(COURSE_DATA)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '')
        courses = [(i + 1, row['course_name'], row['course_description'], row['skills'], row['difficulty'], 
                    'video' if 'video' in row.get('course_type', '').lower() else 'text', row.get('rating', 0))
                   for i, row in df.iterrows()]
    except FileNotFoundError:
        courses = [
            (1, 'Introduction to Python', 'Learn Python', 'python, programming', 'Beginner', 'video', 4.0),
            (2, 'Data Visualization with Tableau', 'Visualize data', 'data science, visualization', 'Intermediate', 'video', 4.5),
            (3, 'Advanced Python', 'Advanced Python concepts', 'python, programming', 'Intermediate', 'text', 4.2),
            (4, 'Machine Learning Basics', 'Intro to ML', 'machine learning, data science', 'Intermediate', 'quiz', 4.0)
        ]
    c.executemany('INSERT OR REPLACE INTO courses VALUES (?, ?, ?, ?, ?, ?, ?)', courses)
    # Interactions
    interactions = [(1, 1, 4, 20, '2025-05-01 22:00:00')]
    c.executemany('INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?)', interactions)
    # Ontology
    ontology = [(2, 1), (3, 1)]
    c.executemany('INSERT OR REPLACE INTO ontology VALUES (?, ?)', ontology)
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect(DATABASE)
