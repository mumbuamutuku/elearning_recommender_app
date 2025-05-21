from flask import render_template, request, jsonify
from services.services import SimpleRecommender
from services.user_service import get_learner_profile, get_context
from models.database import get_db_connection
from utils.crypto import anonymize_learner_id
from datetime import datetime
import pandas as pd

def init_routes(app):
    courses_df = pd.read_csv('coursera.csv')
    courses_df.columns = courses_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '')
    courses_df['content'] = courses_df['course_name'] + ' ' + courses_df['course_description'] + ' ' + courses_df['skills']
    courses_df['content'] = courses_df['content'].fillna('')
    if 'id' not in courses_df.columns:
        courses_df['id'] = courses_df.index.astype(str)
    recommender = SimpleRecommender(courses_df)
    recommender.load_interactions()

    @app.route('/')
    def index():
        sample_courses = courses_df.sample(6).to_dict('records')
        return render_template('index.html', courses=sample_courses)

    @app.route('/recommend', methods=['POST'])
    def recommend():
        if request.content_type.startswith('application/json'):
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()
        user_id = int(data.get('user_id', 0))
        query = data.get('query', '')
        if not user_id:
            return jsonify({'error': 'Invalid user ID'}), 400
        recommendations = recommender.hybrid_recom(user_id, query)
        evaluation = recommender.evaluate_recommendations(user_id, recommendations)
        evaluation = evaluation or {'precision': 0, 'recall': 0, 'diversity': 0, 'novelty': 0}
        evaluation = {k: v * 100 for k, v in evaluation.items()}
        response = {
            'recommendations': recommendations.to_dict('records'),
            'explanation': f"Recommended based on your {'search and ' if query else ''}learning history",
            'evaluation': evaluation
        }
        if request.content_type.startswith('application/json'):
            return jsonify(response)
        return render_template('recommend.html',
                              learner={'id': user_id},
                              context={'time': 'day', 'device': 'desktop'},
                              recommendations=response['recommendations'],
                              evaluation=evaluation,
                              anon_id=f"anon_{user_id}",
                              courses=courses_df.sample(6).to_dict('records'),
                              query=query)

    @app.route('/rate', methods=['POST'])
    def rate_course():
        if request.content_type.startswith('application/json'):
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()
        user_id = int(data.get('learner_id', 0))
        course_id = str(data.get('course_id', '0'))
        rating = int(data.get('rating', 0))
        if not (user_id and course_id and 1 <= rating <= 5):
            return jsonify({'error': 'Invalid input'}), 400
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO interactions VALUES (?, ?, ?, ?, ?)',
                  (user_id, course_id, rating, 10, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        c.execute('INSERT INTO feedback VALUES (?, ?, ?, ?)',
                  (user_id, course_id, rating, data.get('comment', '')))
        conn.commit()
        conn.close()
        recommender.user_interactions[user_id][course_id] = rating
        return jsonify({'status': 'success'})

    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        metrics = recommender.get_overall_metrics()
        return jsonify(metrics) if metrics else jsonify({'message': 'No evaluation data'}), 404

    @app.route('/user/recommendations', methods=['GET', 'POST'])
    def user_recommendations():
        if request.method == 'POST':
            learner_id = int(request.form.get('learner_id', 0))
            if not learner_id:
                return render_template('recommend.html',
                                      courses=courses_df.sample(6).to_dict('records'),
                                      error="Please enter a valid Learner ID")
            try:
                profile = get_learner_profile(learner_id)
                context = get_context(learner_id)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            query = profile['goals']
            recommendations = recommender.hybrid_recom(learner_id, query)
            evaluation = recommender.evaluate_recommendations(learner_id, recommendations)
            evaluation = evaluation or {'precision': 0, 'recall': 0, 'diversity': 0, 'novelty': 0}
            evaluation = {k: v * 100 for k, v in evaluation.items()}
            print(f"Debug: Context: {vars(context) if hasattr(context, '__dict__') else context}")
            print(f"Debug: Recommendations: {recommendations[['id', 'course_name', 'score']].to_dict('records')}")
            print(f"Debug: Evaluation: {evaluation}")
            return render_template('recommend.html',
                                  learner=profile,
                                  context={'time': context.time, 'device': context.device} if hasattr(context, 'time') else context,
                                  recommendations=recommendations.to_dict('records'),
                                  evaluation=evaluation,
                                  anon_id=f"anon_{learner_id}",
                                  courses=courses_df.sample(6).to_dict('records'),
                                  query=query)
        popular_courses = courses_df.sort_values('course_rating', ascending=False).head(6)
        return render_template('recommend.html',
                              courses=popular_courses.to_dict('records'),
                              learner=None,
                              recommendations=None,
                              evaluation=None)
