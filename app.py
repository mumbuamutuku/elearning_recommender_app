from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from collections import defaultdict
import json

app = Flask(__name__)

# Load and preprocess data
def load_data():
    courses = pd.read_csv('data/coursera.csv')
    
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
def init_db():
    conn = sqlite3.connect('learning.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, 
                  username TEXT, 
                  preferences TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (user_id INTEGER,
                  course_id TEXT,
                  rating REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

class SimpleRecommender:
    def __init__(self, courses_df):
        self.courses = courses_df
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
        if user_id not in self.user_interactions or len(self.user_interactions) < 2:
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
            return None
            
        # Get user's positively rated courses (rating >= 3)
        user_positive = {cid for cid, rating in self.user_interactions[user_id].items() if rating >= 3}
        if not user_positive:
            return None
            
        # Get top k recommended course IDs
        recommended = set(recommended_courses['id'].head(k).tolist())
        
        # Calculate precision and recall
        relevant_and_recommended = recommended & user_positive
        precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
        recall = len(relevant_and_recommended) / len(user_positive)
        
        # Calculate diversity (content dissimilarity between recommendations)
        if len(recommended) > 1:
            rec_indices = recommended_courses.head(k).index.tolist()
            submatrix = self.tfidf_matrix[rec_indices]
            pairwise_sim = cosine_similarity(submatrix)
            diversity = 1 - pairwise_sim[np.triu_indices(len(pairwise_sim), k=1)].mean()
        else:
            diversity = 0
            
        # Calculate novelty (how popular are the recommended items)
        popularity = []
        for cid in recommended:
            num_users_rated = sum(1 for u in self.user_interactions.values() if cid in u)
            popularity.append(num_users_rated)
        novelty = 1 - (sum(popularity) / (len(self.user_interactions) * k)) if self.user_interactions else 0
        
        # Update coverage
        self.evaluation_metrics['coverage'].update(recommended)
        
        # Store metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'diversity': diversity,
            'novelty': novelty
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
    
    return jsonify(response)

@app.route('/rate', methods=['POST'])
def rate_course():
    data= request.json
    user_id = data['user_id']
    course_id = data['course_id']
    rating = data['rating']
    
    conn = sqlite3.connect('learning.db')
    c = conn.cursor()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, datetime('now'))", 
              (user_id, course_id, rating))
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


# Add this new route to your existing app.py
@app.route('/user/recommendations', methods=['GET', 'POST'])
def user_recommendations():
    if request.method == 'POST':
        # Handle form submission
        learner_id = int(request.form.get('learner_id', 0))
        query = request.form.get('query', '')
        
        if not learner_id:
            return render_template('index.html', 
                               courses=courses_df.sample(6).to_dict('records'),
                               error="Please enter a valid Learner ID")
        
        # Get recommendations
        recommendations = recommender.hybrid_recommendations(learner_id, query)
        
        # Create a mock user profile (you can replace this with your actual user service)
        profile = {
            'name': f"Learner {learner_id}",
            'age': 25,  # Default age
            'goals': query if query else "General Learning",
            'preferences': "Video",  # Default preference
            'study_hours': 5,  # Default study hours
            'interactions': []  # Placeholder for actual interactions
        }
        
        # Create mock context (replace with your actual context service)
        context = {
            'time': "Daytime",  # Default time
            'device': "Desktop"  # Default device
        }
        
        # Evaluate recommendations
        evaluation = recommender.evaluate_recommendations(learner_id, recommendations)
        
        return render_template('user_recommend.html',
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

if __name__ == '__main__':
    app.run(debug=True)