from models.database import get_learner_feedback, get_interaction_time
from services.recommender import get_recommendations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class UnifiedEvaluator:
    def __init__(self, user_interactions, courses, tfidf_matrix):
        self.user_interactions = user_interactions  # {user_id: {course_id: rating}}
        self.courses = courses                      # DataFrame with course info
        self.tfidf_matrix = tfidf_matrix            # Course TF-IDF features
        self.evaluation_metrics = {
            'precision': [],
            'recall': [],
            'diversity': [],
            'novelty': [],
            'coverage': set(),
            'engagement': []
        }

    def evaluate_user_performance(self, user_id, k=5):
        recommendations = get_recommendations(user_id)
        if not recommendations:
            return None

        recommended_ids = [rec.course_id for rec in recommendations[:k] if rec.course_id]
        recommended_set = set(recommended_ids)

        # Feedback-based precision
        feedback = get_learner_feedback(user_id)
        relevant = {cid for cid, rating in feedback if rating >= 4}
        hits = len(recommended_set & relevant)
        precision = hits / k if k else 0

        # Engagement
        engagement = get_interaction_time(user_id, recommended_ids)

        # Recall (using stored interactions)
        user_positive = {cid for cid, rating in self.user_interactions.get(user_id, {}).items() if rating >= 3}
        recall = len(recommended_set & user_positive) / len(user_positive) if user_positive else 0

        # Diversity
        rec_indices = self.courses[self.courses['id'].isin(recommended_set)].index.tolist()
        if len(rec_indices) > 1:
            submatrix = self.tfidf_matrix[rec_indices]
            pairwise_sim = cosine_similarity(submatrix)
            diversity = 1 - pairwise_sim[np.triu_indices(len(pairwise_sim), k=1)].mean()
        else:
            diversity = 0

        # Novelty
        popularity = [sum(1 for u in self.user_interactions.values() if cid in u) for cid in recommended_set]
        novelty = 1 - (sum(popularity) / (len(self.user_interactions) * k)) if self.user_interactions else 0

        # Coverage update
        self.evaluation_metrics['coverage'].update(recommended_set)

        # Store metrics
        self.evaluation_metrics['precision'].append(precision)
        self.evaluation_metrics['recall'].append(recall)
        self.evaluation_metrics['diversity'].append(diversity)
        self.evaluation_metrics['novelty'].append(novelty)
        self.evaluation_metrics['engagement'].append(engagement)

        return {
            'user_id': user_id,
            'precision': precision,
            'recall': recall,
            'diversity': diversity,
            'novelty': novelty,
            'engagement': engagement
        }

    def evaluate_all_users(self):
        results = []
        for user_id in self.user_interactions:
            user_metrics = self.evaluate_user_performance(user_id)
            if user_metrics:
                results.append(user_metrics)
        return results

    def get_average_metrics(self):
        return {
            'avg_precision': np.mean(self.evaluation_metrics['precision']) if self.evaluation_metrics['precision'] else 0,
            'avg_recall': np.mean(self.evaluation_metrics['recall']) if self.evaluation_metrics['recall'] else 0,
            'avg_diversity': np.mean(self.evaluation_metrics['diversity']) if self.evaluation_metrics['diversity'] else 0,
            'avg_novelty': np.mean(self.evaluation_metrics['novelty']) if self.evaluation_metrics['novelty'] else 0,
            'avg_engagement': np.mean(self.evaluation_metrics['engagement']) if self.evaluation_metrics['engagement'] else 0,
            'coverage': len(self.evaluation_metrics['coverage'])
        }
