from models.database import get_db_connection

def get_prerequisites(course_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT prerequisite_id FROM ontology WHERE course_id = ?', (course_id,))
    prereqs = [row[0] for row in c.fetchall()]
    conn.close()
    return prereqs
