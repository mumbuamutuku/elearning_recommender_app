from datetime import datetime
from services.prerequisite_service import get_prerequisite_id
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from collections import defaultdict
import json
from models.database import populate_sample_data, init_db

from services.evaluation import UnifiedEvaluator
from services.user_service import get_context, get_learner_profile

app = Flask(__name__)

# Load and preprocess data
def load_data():
    courses = pd.read_csv('coursera.csv')
    
    # Clean column names (remove special characters)
    courses.columns = courses.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '')
    
    # Create content for TF-IDF
    courses['content'] = courses['course_name'] + ' ' + courses['course_description'] + ' ' + courses['skills']
    courses['content'] = courses['content'].fillna('')
    
    # Ensure course_id exists - we'll use index if not
    if 'id' not in courses.columns:
        courses['id'] = courses.index.astype(str)
    
    return courses

courses_df = load_data()

# Initialize SQLite database
# def init_db():
#     conn = sqlite3.connect('learning.db')
#     c = conn.cursor()
    
#     c.execute('''CREATE TABLE IF NOT EXISTS users
#                  (id INTEGER PRIMARY KEY, 
#                   name TEXT, 
#                   age INTEGER, 
#                   goals TEXT, 
#                   preferences TEXT)''')
    
#     c.execute('''CREATE TABLE IF NOT EXISTS interactions
#                  (user_id INTEGER,
#                   course_id TEXT,
#                   rating REAL,
#                   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
#     conn.commit()
#     conn.close()

# init_db()

class SimpleRecommender:
    def __init__(self, courses_df):
        self.courses = courses_df
        self.courses['id'] = self.courses['id'].astype(str)
        # self.courses['course_rating'] = self.courses.get('course_rating', 0).astype(float)
        self.user_interactions = defaultdict(dict)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.courses['content'])
        self.evaluation_metrics = {
            'precision': [],
            'recall': [],
            'diversity': [],
            'novelty': [],
            'coverage': set()
        }
        
    def load_interactions(self):
        conn = sqlite3.connect('learning.db')
        interactions = pd.read_sql("SELECT * FROM interactions", conn)
        conn.close()
        
        for _, row in interactions.iterrows():
            self.user_interactions[row['user_id']][row['course_id']] = row['rating']
    
    def content_based_recommendations(self, query, n=5):
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = (-similarities).argsort()[:n]
        return self.courses.iloc[related_indices]
    
    def collaborative_recommendations(self, user_id, n=5):
        if user_id not in self.user_interactions or len(self.user_interactions[user_id]) == 0:
        # if user_id not in self.user_interactions or len(self.user_interactions) < 2:
            return pd.DataFrame()
            
        # Simple user-user collaborative filtering
        user_ratings = self.user_interactions[user_id]
        similarities = []
        
        for other_user, ratings in self.user_interactions.items():
            if other_user == user_id:
                continue
                
            # Calculate cosine similarity between users
            common_courses = set(user_ratings.keys()) & set(ratings.keys())
            if not common_courses:
                continue
                
            vec1 = np.array([user_ratings[c] for c in common_courses])
            vec2 = np.array([ratings[c] for c in common_courses])
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append((similarity, other_user))
        
        if not similarities:
            return pd.DataFrame()
            
        # Get top similar users
        similarities.sort(reverse=True)
        top_users = [user for _, user in similarities[:3]]
        
        # Recommend courses liked by similar users
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
    
    def evaluate_recommendations(self, user_id, recommended_courses, k=5):
        """
        Evaluate recommendations against user's actual interactions
        """
        if user_id not in self.user_interactions:
            # print(f"Debug: User {user_id} has no interactions")
            return None
            
        # Get user's positively rated courses (rating >= 3)
        user_positive = {cid for cid, rating in self.user_interactions[user_id].items() if rating >= 3}
        if not user_positive:
            # print(f"Debug: User {user_id} has no positive ratings (>=3)")
            return None
            
        # Get top k recommended course IDs
        # recommended = set(recommended_courses['id'].head(k).tolist())
        recommended = set(str(cid) for cid in recommended_courses['id'].head(k).tolist())
        # print(f"Debug: Recommended IDs: {recommended}")
        # print(f"Debug: User positive IDs: {user_positive}")

        # Calculate precision and recall
        relevant_and_recommended = recommended & user_positive
        # print(f"Debug: Relevant and recommended: {relevant_and_recommended}")
        precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
        recall = len(relevant_and_recommended) / len(user_positive)
        
        # Calculate diversity (content dissimilarity between recommendations)
        if len(recommended) > 1:
            rec_indices = self.courses[self.courses['id'].isin(recommended)].index.tolist()
            # rec_indices = recommended_courses.head(k).index.tolist()
            # submatrix = self.tfidf_matrix[rec_indices]
            # pairwise_sim = cosine_similarity(submatrix)
            # diversity = 1 - pairwise_sim[np.triu_indices(len(pairwise_sim), k=1)].mean()
            if len(rec_indices) > 1:
                submatrix = self.tfidf_matrix[rec_indices]
                pairwise_sim = cosine_similarity(submatrix)
                diversity = 1 - pairwise_sim[np.triu_indices(len(pairwise_sim), k=1)].mean()
            else:
                diversity = 0
        else:
            diversity = 0
            
        # Calculate novelty (how popular are the recommended items)
        popularity = []
        for cid in recommended:
            num_users_rated = sum(1 for u in self.user_interactions.values() if cid in u)
            popularity.append(num_users_rated)
        # novelty = 1 - (sum(popularity) / (len(self.user_interactions) * k)) if self.user_interactions else 0
        total_users = max(1, len(self.user_interactions))
        novelty = 1 - (sum(popularity) / (total_users * k)) if k > 0 else 0
        
        # Update coverage
        self.evaluation_metrics['coverage'].update(recommended)
        
        # Store metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'diversity': diversity,
            'novelty': max(novelty, 0.1),
        }
        
        for metric, value in metrics.items():
            self.evaluation_metrics[metric].append(value)
            
        return metrics
    
    def get_overall_metrics(self):
        """Return aggregated metrics over all evaluations"""
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

    def _generate_explanation(self, course_idx, course_id, query, 
                            content_weight, collab_weight, 
                            completed_courses, cb_indices, cf_indices):
        """Generate personalized explanation for a recommendation"""
        # Check prerequisites
        prerequisites = get_prerequisite_id(course_id)
        prerequisites_met = all(p in completed_courses for p in prerequisites)
        
        # Determine recommendation source
        from_cb = course_idx in cb_indices
        from_cf = course_id in cf_indices
        
        # Build explanation components
        explanation_parts = []
        
        if prerequisites_met and prerequisites:
            explanation_parts.append(
                f"Recommended based on your goal '{query}' and completed prerequisites"
            )
            
        if from_cb and from_cf:
            explanation_parts.append(
                "recommended through both content analysis and peer preferences"
            )
        elif from_cb:
            explanation_parts.append(
                "recommended based on course content matching your goals" 
                if content_weight > collab_weight else
                "recommended through content analysis with collaborative insights"
            )
        elif from_cf:
            explanation_parts.append(
                "recommended by learners with similar interests"
                if collab_weight > content_weight else
                "recommended through peer preferences with content consideration"
            )
            
        if not explanation_parts:
            explanation_parts.append("recommended as a popular choice among learners")
            
        # Add knowledge graph reference if available
        # if prerequisites:
        #     top_prerequisites = ', '.join(prerequisites[:3])
        #     explanation_parts.append(
        #         f"related to: {top_prerequisites} knowledge domains"
        #     )
            
        # Capitalize first letter and add period
        explanation = ', '.join(explanation_parts) + '.'
        return explanation[0].upper() + explanation[1:]
    
    
    def hybrid_recom(self, user_id, query=None, n=5):
        # Get profile maturity for weighting
        profile = get_learner_profile(user_id)
        profile_maturity = len(self.user_interactions.get(user_id, {}))
        collab_weight = 0.7 if profile_maturity > 2 else 0.2
        content_weight = 0.3 if profile_maturity > 2 else 0.8
        context = get_context(user_id)
        # Get base recommendations
        cb_recs = self.content_based_recommendations(query or profile['goals'], n) if query or profile['goals'] else pd.DataFrame()
        cf_recs = self.collaborative_recommendations(user_id, n)
        
        # Get indices for explanation generation
        cb_indices = set(cb_recs.index.tolist()) if not cb_recs.empty else set()
        cf_course_ids = set(cf_recs['id'].tolist()) if not cf_recs.empty else set()
        
        # Combine recommendations
        # combined = pd.concat([cb_recs, cf_recs]).drop_duplicates().head(n)
        if not cb_recs.empty and not cf_recs.empty:
            combined = pd.concat([cb_recs, cf_recs]).drop_duplicates().head(n)
        elif not cb_recs.empty:
            combined = cb_recs.head(n)
        elif not cf_recs.empty:
            combined = cf_recs.head(n)
        else:
            combined = self.courses.sample(n)
        
        # Get completed courses for prerequisite check
        completed_courses = set(self.user_interactions.get(user_id, {}).keys())
        
        # Generate explanations for each course
        # explanations = []
        # for idx, course in combined.iterrows():
        #     explanation = self._generate_explanation(
        #         idx, course['id'], query or "learning goals",
        #         content_weight, collab_weight, completed_courses,
        #         cb_indices, cf_course_ids
        #     )
        #     explanations.append(explanation)
            
        # combined = combined.copy()
        # combined['explanation'] = explanations
        
        # return combined
        recommendations = []
        for idx, course in combined.iterrows():
            score = (collab_weight * course.get('rating', 0) + content_weight * cosine_similarity(
                self.tfidf_vectorizer.transform([query or profile['goals']]),
                self.tfidf_matrix[idx])[0][0]) / (collab_weight + content_weight)
            context_device = context.device if hasattr(context, 'device') else context.get('device', 'desktop')
            context_time = context.time if hasattr(context, 'time') else context.get('time', 'day')
            if context_device == 'mobile' and context_time == 'night' and course.get('format', '') == 'video':
                score *= 1.2
            if get_prerequisite_id(course['id']) and all(p in completed_courses for p in get_prerequisite_id(course['id'])):
                score += 0.2
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
    
recommender = SimpleRecommender(courses_df)
recommender.load_interactions()

@app.route('/')
def home():
    # Convert sample courses to dictionary records
    sample_courses = courses_df.sample(6).to_dict('records')
    return render_template('index.html', courses=sample_courses)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    query = data.get('query', '')
    
    recommendations = recommender.hybrid_recommendations(user_id, query)
    recommendations_dict = recommendations.to_dict('records')
    
    # Evaluate the recommendations
    evaluation = recommender.evaluate_recommendations(user_id, recommendations)
    
    response = {
        'recommendations': recommendations_dict,
        'explanation': f"Recommended based on your {'search and ' if query else ''}learning history"
    }
    
    if evaluation:
        response['evaluation'] = evaluation

    # print(response)
    
    return jsonify(response)


@app.route('/rate', methods=['POST'])
def rate_course():
    data= request.json
    user_id = data['user_id']
    course_id = data['course_id']
    rating = data['rating']
    comment = data['comment']
    
    conn = sqlite3.connect('learning.db')
    c = conn.cursor()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, datetime('now'))", 
              (user_id, course_id, rating, comment))
    conn.commit()
    conn.close()
    
    # Update recommender's interactions
    recommender.user_interactions[user_id][course_id] = rating
    return jsonify({'status': 'success'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = recommender.get_overall_metrics()
    if metrics:
        return jsonify(metrics)
    else:
        return jsonify({'message': 'No evaluation data available yet'}), 404


@app.route('/user/recommendations', methods=['GET', 'POST'])
def user_recommendations():
    if request.method == 'POST':
        # Handle form submission
        learner_id = int(request.form.get('learner_id', 0))
        # print(learner_id)

        if not learner_id:
            return render_template('recommend.html', 
                               courses=courses_df.sample(6).to_dict('records'),
                               error="Please enter a valid Learner ID")

        # Get learner profile & context
        try:
            profile = get_learner_profile(learner_id)
            context = get_context(learner_id)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        # Get recommendations
        #query = request.form.get('query', '')
        query = profile['goals']
               
        # Get recommendations
        recommendations = recommender.hybrid_recom(learner_id, query)

        # Evaluate recommendations
        evaluation = recommender.evaluate_recommendations(learner_id, recommendations)
        # user_interactions = recommender.user_interactions
        # tfidf_matrix = recommender.tfidf_matrix
        # evaluator = UnifiedEvaluator(user_interactions, courses_df, tfidf_matrix)
        # evaluation = evaluator.evaluate_user_performance(user_id=learner_id)
        
        return render_template('recommend.html',
                           learner=profile,
                           context=context,
                           recommendations=recommendations.to_dict('records'),
                           evaluation=evaluation,
                           anon_id=f"anon_{learner_id}",
                           courses=courses_df.sample(6).to_dict('records'),
                           query=query)
    
    # For GET requests, show the form with popular courses
    popular_courses = courses_df.sort_values('course_rating', ascending=False).head(6)
    return render_template('user_recommend.html',
                       courses=popular_courses.to_dict('records'),
                       learner=None,
                       recommendations=None,
                       evaluation=None)

@app.route('/feedback', methods=['POST'])
def rate_courses():
    # Handle form or JSON data
    if request.content_type.startswith('application/json'):
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    learner_id = int(data.get('learner_id', 0))
    course_id = int(data.get('course_id', 0))
    rating = int(data.get('rating', 0))
    comment = data.get('comment', '')

    if not (learner_id and course_id and 1 <= rating <= 5):
        return jsonify({'error': 'Invalid input'}), 400

    conn = sqlite3.connect('learning.db')
    c = conn.cursor()
    c.execute('INSERT INTO interactions VALUES (?, ?, ?,  ?)',
                (learner_id, course_id, rating,  datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    c.execute('INSERT INTO feedback VALUES (?, ?, ?, ?)',
                (learner_id, course_id, rating, comment))
    c.execute('SELECT AVG(rating) FROM feedback WHERE course_id = ?', (course_id,))
    avg_rating = c.fetchone()[0] or rating
    c.execute('UPDATE interactions SET rating = ? WHERE course_id = ?', (avg_rating, course_id))
    conn.commit()
    conn.close()

    recommender.user_interactions[learner_id][course_id] = rating
    return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})


# if __name__ == '__main__':
#     # Create the database and tables
#     init_db()
        
#     # Add sample data
#     populate_sample_data()

#     # app.run(debug=True)
#     app.run()