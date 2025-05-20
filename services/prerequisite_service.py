from models.database import get_db_connection

def get_prerequisite_id(goal_course_ids):
    """
    Return a set of course IDs that are prerequisites for the learner's goal-aligned courses.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Collect all prerequisite course IDs
    placeholders = ",".join("?" for _ in goal_course_ids)
    query = f"SELECT prerequisite_id FROM prerequisites WHERE course_id IN ({placeholders})"

    c.execute(query, goal_course_ids)
    prerequisites = sorted({row[0] for row in c.fetchall()})
    
    conn.close()
    return prerequisites
