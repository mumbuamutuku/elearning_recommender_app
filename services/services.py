import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from models.database import get_db_connection
from services.user_service import get_learner_profile, get_context
from services.prerequisite_service import get_prerequisite_id
from collections import defaultdict

class SimpleRecommender:
    def __init__(self, courses_df):
        self.courses = courses_df
        self.courses['id'] = self.courses['id'].astype(str)
        # self.courses['course_rating'] = self.courses.get('course_rating', 0).astype(float)
        self.user_interactions = defaultdict(dict)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.courses['content'])
        self.evaluation_metrics = {'precision': [], 'recall': [], 'diversity': [], 'novelty': [], 'coverage': set()}

    def load_interactions(self):
        conn = get_db_connection()
        interactions = pd.read_sql('SELECT * FROM interactions', conn)
        conn.close()
        for _, row in interactions.iterrows():
            self.user_interactions[row['learner_id']][str(row['course_id'])] = row['rating']

    def content_based_recommendations(self, query, n=5):
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = (-similarities).argsort()[:n]
        return self.courses.iloc[related_indices]

    def collaborative_recommendations(self, user_id, n=5):
        if user_id not in self.user_interactions or len(self.user_interactions[user_id]) == 0:
            return pd.DataFrame()
        user_ratings = self.user_interactions[user_id]
        similarities = []
        for other_user, ratings in self.user_interactions.items():
            if other_user == user_id:
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

    def hybrid_recommendations(self, user_id, query=None, n=5):
        cb_recs = self.content_based_recommendations(query, n) if query else pd.DataFrame()
        cf_recs = self.collaborative_recommendations(user_id, n)
        if not cb_recs.empty and not cf_recs.empty:
            combined = pd.concat([cb_recs, cf_recs]).drop_duplicates()
            return combined.head(n)
        elif not cb_recs.empty:
            return cb_recs
        else:
            return cf_recs if not cf_recs.empty else self.courses.sample(n)

    def hybrid_recom(self, user_id, query=None, n=5):
        profile = get_learner_profile(user_id)
        profile_maturity = len(self.user_interactions.get(user_id, {}))
        collab_weight = 0.7 if profile_maturity > 2 else 0.2
        content_weight = 0.3 if profile_maturity > 2 else 0.8
        context = get_context(user_id)
        cb_recs = self.content_based_recommendations(query or profile['goals'], n) if query or profile['goals'] else pd.DataFrame()
        cf_recs = self.collaborative_recommendations(user_id, n)
        cb_indices = set(cb_recs.index.tolist()) if not cb_recs.empty else set()
        cf_course_ids = set(cf_recs['id'].tolist()) if not cf_recs.empty else set()
        if not cb_recs.empty and not cf_recs.empty:
            combined = pd.concat([cb_recs, cf_recs]).drop_duplicates().head(n)
        elif not cb_recs.empty:
            combined = cb_recs.head(n)
        elif not cf_recs.empty:
            combined = cf_recs.head(n)
        else:
            combined = self.courses.sample(n)
        completed_courses = set(self.user_interactions.get(user_id, {}).keys())
        recommendations = []
        for idx, course in combined.iterrows():
            score = (collab_weight * course.get('course_rating', 0) + content_weight * cosine_similarity(
                self.tfidf_vectorizer.transform([query or profile['goals']]),
                self.tfidf_matrix[idx])[0][0]) / (collab_weight + content_weight)
            # Handle Context object or dictionary
            context_device = context.device if hasattr(context, 'device') else context.get('device', 'desktop')
            context_time = context.time if hasattr(context, 'time') else context.get('time', 'day')
            if context_device == 'mobile' and context_time == 'night' and course.get('format', '') == 'video':
                score *= 1.2
            if get_prerequisite_id(course['id']) and all(p in completed_courses for p in get_prerequisite_id(course['id'])):
                score += 0.2
            explanation = self._generate_explanation(
                idx, course['id'], query or profile['goals'],
                content_weight, collab_weight, completed_courses,
                cb_indices, cf_course_ids
            )
            recommendations.append({
                'id': course['id'],
                'course_name': course.get('course_name', course.get('title', 'Unknown')),
                'university': course.get('university', 'Unknown'),
                'course_description': course.get('course_description', 'No description available'),
                'difficulty_level': course.get('difficulty', course.get('difficulty_level', 'Unknown')),
                'course_rating': course.get('course_rating', 0),
                'skills': course.get('skills', ''),
                'course_url': course.get('course_url', '#'),
                'explanation': explanation,
                'score': score
            })
        result = pd.DataFrame(recommendations)
        return result.sort_values('score', ascending=False).head(n)

    def evaluate_recommendations(self, user_id, recommended_courses, k=5):
        if user_id not in self.user_interactions:
            print(f"Debug: User {user_id} has no interactions")
            return None
        user_positive = {str(cid) for cid, rating in self.user_interactions[user_id].items() if rating >= 3}
        if not user_positive:
            print(f"Debug: User {user_id} has no positive ratings (>=3)")
            return None
        recommended = set(str(cid) for cid in recommended_courses['id'].head(k).tolist())
        print(f"Debug: Recommended IDs: {recommended}")
        print(f"Debug: User positive IDs: {user_positive}")
        relevant_and_recommended = recommended & user_positive
        print(f"Debug: Relevant and recommended: {relevant_and_recommended}")
        precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
        recall = len(relevant_and_recommended) / len(user_positive) if user_positive else 0
        if len(recommended) > 1:
            rec_indices = self.courses[self.courses['id'].isin(recommended)].index.tolist()
            if len(rec_indices) > 1:
                submatrix = self.tfidf_matrix[rec_indices]
                pairwise_sim = cosine_similarity(submatrix)
                diversity = 1 - pairwise_sim[np.triu_indices(len(pairwise_sim), k=1)].mean()
            else:
                diversity = 0
        else:
            diversity = 0
        popularity = []
        for cid in recommended:
            num_users_rated = sum(1 for u in self.user_interactions.values() if str(cid) in u)
            popularity.append(num_users_rated)
        total_users = max(1, len(self.user_interactions))
        novelty = 1 - (sum(popularity) / (total_users * k)) if k > 0 else 0
        self.evaluation_metrics['coverage'].update(recommended)
        metrics = {
            'precision': precision,
            'recall': recall,
            'diversity': diversity,
            'novelty': max(novelty, 0.1),
        }
        print(f"Debug: Metrics: {metrics}")
        for metric, value in metrics.items():
            self.evaluation_metrics[metric].append(value)
        return metrics

    def get_overall_metrics(self):
        if not any(self.evaluation_metrics.values()):
            return None
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

    def _generate_explanation(self, course_idx, course_id, query, content_weight, collab_weight, completed_courses, cb_indices, cf_indices):
        prerequisites = get_prerequisite_id(course_id)
        prerequisites_met = all(p in completed_courses for p in prerequisites)
        from_cb = course_idx in cb_indices
        from_cf = course_id in cf_indices
        explanation_parts = []
        if prerequisites_met and prerequisites:
            explanation_parts.append(f"matches your goal '{query}' and completed prerequisites")
        if from_cb and from_cf:
            explanation_parts.append("based on both content analysis and peer preferences")
        elif from_cb:
            explanation_parts.append(
                "based on course content matching your goals"
                if content_weight > collab_weight else
                "through content analysis with collaborative insights"
            )
        elif from_cf:
            explanation_parts.append(
                "by learners with similar interests"
                if collab_weight > content_weight else
                "through peer preferences with content consideration"
            )
        if not explanation_parts:
            explanation_parts.append("as a popular choice among learners")
        explanation = ', '.join(explanation_parts) + '.'
        return explanation[0].upper() + explanation[1:]
