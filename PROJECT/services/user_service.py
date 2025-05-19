from models.database import get_db_connection
from datetime import datetime
import random

def get_learner_profile(learner_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM learners WHERE id = ?', (learner_id,))
    profile = c.fetchone()
    c.execute('SELECT course_id, rating, time_spent FROM interactions WHERE learner_id = ?', (learner_id,))
    interactions = c.fetchall()
    conn.close()
    return {
        'id': profile[0], 'name': profile[1], 'age': profile[2],
        'goals': profile[3], 'preferences': profile[4], 'study_hours': profile[5],
        'interactions': interactions
    }

def get_context(learner_id):
    current_time = datetime.now().hour
    device = 'mobile' if random.random() > 0.5 else 'desktop'
    return {'time': 'night' if current_time >= 18 else 'day', 'device': device}
