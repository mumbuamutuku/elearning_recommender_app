import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from models.database import get_db_connection
from services.user_service import get_learner_profile, get_context
from services.ontology_service import get_prerequisites
from collections import defaultdict

class SimpleRecommender:
    def __init__(self):
        conn = get_db_connection()
        self.courses = pd.read_sql('SELECT * FROM courses', conn)
        self.courses['content'] = self.courses['title'] + ' ' + self.courses['description'] + ' ' + self.courses['skills']
        self.courses['content'] = self.courses['content'].fillna('')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.courses['content'])
        self.user_interactions = defaultdict(dict)
        self.evaluation_metrics = {'precision': [], 'recall': [], 'diversity': [], 'novelty': [], 'coverage': set()}
        conn.close()

    def load_interactions(self):
        conn = get_db_connection()
        interactions = pd.read_sql('SELECT * FROM interactions', conn)
        conn.close()
        for _, row in interactions.iterrows():
            self.user_interactions[row['learner_id']][row['course_id']] = row['rating']

    def content_based_recommendations(self, query, n=5):
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = (-similarities).argsort()[:n]
        return self.courses.iloc[related_indices]

    def collaborative_recommendations(self, learner_id, n=5):
        if learner_id not in self.user_interactions or len(self.user_interactions) < 2:
            return pd.DataFrame()
        user_ratings = self.user_interactions[learner_id]
        similarities = []
        for other_user, ratings in self.user_interactions.items():
            if other_user == learner_id:
                continue
            common_courses = set(user_ratings.keys()) & set(ratings.keys())
            if not common_courses:
                continue
            vec1 = np.array([user_ratings[c] for c in common_courses])
            vec2 = np.array([ratings[c] for c in common_courses])
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append((similarity, other_user))
        if not similarities:
            return pd.DataFrame()
        similarities.sort(reverse=True)
        top_users = [user for _, user in similarities[:3]]
        recommended_courses = set()
        for user in top_users:
            for course_id, rating in self.user_interactions[user].items():
                if rating >= 3.5 and course_id not in user_ratings:
                    recommended_courses.add(course_id)
        return self.courses[self.courses['id'].isin(list(recommended_courses)[:n])]

    def hybrid_recommendations(self, learner_id, query=None, n=5):
        profile = get_learner_profile(learner_id)
        profile_maturity = len(profile['interactions'])
        collab_weight = 0.7 if profile_maturity > 2 else 0.2
        content_weight = 0.3 if profile_maturity > 2 else 0.8
        context = get_context(learner_id)
        cb_recs = self.content_based_recommendations(query or profile['goals'], n) if query or profile['goals'] else pd.DataFrame()
        cf_recs = self.collaborative_recommendations(learner_id, n)
        
        # Combine recommendations
        if not cb_recs.empty and not cf_recs.empty:
            combined = pd.concat([cb_recs, cf_recs]).drop_duplicates()
        elif not cb_recs.empty:
            combined = cb_recs
        elif not cf_recs.empty:
            combined = cf_recs
        else:
            combined = self.courses.sample(n)
        
        recommendations = []
        completed_courses = [i[0] for i in profile['interactions']]
        for _, course in combined.iterrows():
            score = (collab_weight * course.get('rating', 0) + content_weight * cosine_similarity(
                self.tfidf_vectorizer.transform([profile['goals']]),
                self.tfidf_matrix[course.name])[0][0]) / (collab_weight + content_weight)
            if context['device'] == 'mobile' and context['time'] == 'night' and course['format'] == 'video':
                score *= 1.2
            if get_prerequisites(course['id']) and all(p in completed_courses for p in get_prerequisites(course['id'])):
                score += 0.2
            explanation = f"Recommended for your {profile['goals']} goal"
            if get_prerequisites(course['id']) and all(p in completed_courses for p in get_prerequisites(course['id'])):
                explanation += " and completed prerequisites"
            if profile['preferences'] == course['format']:
                explanation += f" and {profile['preferences']} preference"
            recommendations.append((course['title'], score, explanation))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(recommendations[:n], columns=['title', 'score', 'explanation'])

    def evaluate_recommendations(self, learner_id, recommended_courses, k=5):
        if learner_id not in self.user_interactions:
            return None
        user_positive = {cid for cid, rating in self.user_interactions[learner_id].items() if rating >= 4}
        if not user_positive:
            return None
        recommended = set(recommended_courses['title'].head(k).tolist())
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT title FROM courses WHERE id IN ({})'.format(','.join('?' * len(user_positive))), list(user_positive))
        positive_titles = set(row[0] for row in c.fetchall())
        relevant_and_recommended = recommended & positive_titles
        precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
        recall = len(relevant_and_recommended) / len(positive_titles) if positive_titles else 0
        if len(recommended) > 1:
            rec_indices = [self.courses[self.courses['title'] == t].index[0] for t in recommended if t in self.courses['title'].values]
            submatrix = self.tfidf_matrix[rec_indices]
            pairwise_sim = cosine_similarity(submatrix)
            diversity = 1 - pairwise_sim[np.triu_indices(len(pairwise_sim), k=1)].mean() if len(rec_indices) > 1 else 0
        else:
            diversity = 0
        popularity = []
        for title in recommended:
            c.execute('SELECT COUNT(*) FROM interactions WHERE course_id IN (SELECT id FROM courses WHERE title = ?)', (title,))
            num_users_rated = c.fetchone()[0]
            popularity.append(num_users_rated)
        novelty = 1 - (sum(popularity) / (len(self.user_interactions) * k)) if self.user_interactions else 0
        self.evaluation_metrics['coverage'].update(recommended)
        c.execute('SELECT AVG(time_spent) FROM interactions WHERE learner_id = ? AND course_id IN (SELECT id FROM courses WHERE title IN ({}) )'.format(
            ','.join('?' * len(recommended))), [learner_id] + list(recommended))
        # engagement = c.fetchone()[0] or 0
        conn.close()
        metrics = {
            'precision': precision,
            'recall': recall,
            'diversity': diversity,
            'novelty': novelty,
            # 'engagement': engagement
        }
        for metric, value in metrics.items():
            if metric != 'coverage':
                self.evaluation_metrics[metric].append(value)
        return metrics

    def get_overall_metrics(self):
        aggregated = {}
        for metric, values in self.evaluation_metrics.items():
            if metric == 'coverage':
                aggregated[metric] = {
                    'absolute': len(values),
                    'relative': len(values) / len(self.courses) if len(self.courses) > 0 else 0
                }
            elif values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        return aggregated
