"""
User profile and context management
"""

import random
from datetime import datetime
from models.database import get_db_connection
from models.schemas import Learner, Context

def get_learner_profile(learner_id):
    """
    Retrieve a complete learner profile including interactions
    
    Args:
        learner_id: The ID of the learner
        
    Returns:
        Learner: A learner object with all profile data
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get basic profile data
    c.execute('SELECT * FROM learners WHERE id = ?', (learner_id,))
    profile = c.fetchone()
        
    # Get learner's interactions
    c.execute('SELECT course_id, rating FROM interactions WHERE user_id = ?', (learner_id,))
    interactions = c.fetchall()
    
    conn.close()
    
    # Create and return a Learner object
    # return Learner.from_db_row(profile, interactions)
    if not profile:
            raise ValueError(f"Learner {learner_id} not found")
    print(f"Debug: Profile tuple for learner {learner_id}: {profile}, length: {len(profile)}")
    return {
        'id': profile[0],
        'name': profile[1],
        'age': profile[2],
        'goals': profile[3],
        'preferences': profile[4],
        'interactions': interactions
    }
def get_context(learner_id):
    """
    Determine user context (time of day, device type)
    In a real system, this would detect actual user context.
    
    Args:
        learner_id: ID of the learner (not used in this simulation)
        
    Returns:
        Context: User context information
    """
    # Simulate context - in a real system, this would be determined from actual user data
    current_time = datetime.now().hour
    device = 'mobile' if random.random() > 0.5 else 'desktop'
    time_of_day = 'night' if current_time >= 18 else 'day'
    return {'time': time_of_day, 'device': device} # Context(time=time_of_day, device=device)


def has_sufficient_history(learner_id, threshold=3):
    """
    Check if learner has sufficient interaction history
    
    Args:
        learner_id: The learner's ID
        threshold: Minimum number of interactions to be considered sufficient
        
    Returns:
        bool: True if learner has sufficient history, False otherwise
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM interactions WHERE user_id = ?', (learner_id,))
    count = c.fetchone()[0]
    conn.close()
    
    return count >= threshold