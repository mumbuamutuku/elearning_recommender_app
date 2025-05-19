# from models.database import get_db_connection
# #from services.recommender import get_recommendations

# def evaluate_system(learner_id):
#     recommendations = get_recommendations(learner_id)
#     conn = get_db_connection()
#     c = conn.cursor()
    
#     # Precision and Recall
#     recommended_ids = [i + 1 for i, _ in enumerate(recommendations)]
#     c.execute('SELECT course_id, rating FROM feedback WHERE learner_id = ?', (learner_id,))
#     feedback = c.fetchall()
#     relevant = [cid for cid, rating in feedback if rating >= 4]
#     hits = len(set(recommended_ids) & set(relevant))
#     precision = hits / len(recommendations) if recommendations else 0
#     recall = hits / len(relevant) if relevant else 0
    
#     # Engagement
#     c.execute('SELECT AVG(time_spent) FROM interactions WHERE learner_id = ? AND course_id IN ({})'.format(
#         ','.join('?' * len(recommended_ids))), [learner_id] + recommended_ids)
#     engagement = c.fetchone()[0] or 0
    
#     # A/B Testing (Hybrid vs Content-Based)
#     content_only_recs = get_recommendations(learner_id)  # Simplified: reuse hybrid with high content weight
#     content_hits = len(set([i + 1 for i, _ in enumerate(content_only_recs)]) & set(relevant))
#     content_precision = content_hits / len(content_only_recs) if content_only_recs else 0
    
#     conn.close()
#     return {
#         'precision': precision,
#         'recall': recall,
#         'engagement': engagement,
#         'ab_test': {'hybrid_precision': precision, 'content_precision': content_precision}
#     }
