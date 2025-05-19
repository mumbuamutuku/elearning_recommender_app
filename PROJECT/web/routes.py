from flask import render_template, request, jsonify
from services.recommender import SimpleRecommender
from services.user_service import get_learner_profile, get_context
# from services.evaluation import evaluate_system
from utils.crypto import anonymize_learner_id
from models.database import get_db_connection
from datetime import datetime

def init_routes(app):
    recommender = SimpleRecommender()
    recommender.load_interactions()

    @app.route('/')
    def index():
        try:
            # Get the minimum between 6 and the number of available courses
            sample_size = min(6, len(recommender.courses))
            
            if sample_size > 0:
                # Use replace=True if sampling same size as population
                replace = sample_size >= len(recommender.courses)
                sample_courses = recommender.courses.sample(sample_size, replace=replace).to_dict('records')
            else:
                sample_courses = []
                
            return render_template('index.html', courses=sample_courses)
            
        except Exception as e:
            print(f"Error sampling courses: {str(e)}")
            return render_template('index.html', courses=[])

    @app.route('/recommend', methods=['POST'])
    def recommend():
        # Handle form or JSON data
        if request.content_type.startswith('application/json'):
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()

        learner_id = int(data.get('learner_id', 0))
        query = data.get('query', '')
        anon_id = anonymize_learner_id(learner_id)

        if not learner_id:
            return jsonify({'error': 'Invalid learner ID'}), 400

        recommendations = recommender.hybrid_recommendations(learner_id, query)
        profile = get_learner_profile(learner_id)
        context = get_context(learner_id)
        evaluation = recommender.evaluate_recommendations(learner_id, recommendations)

        # Render HTML for form submissions, JSON for API calls
        if request.content_type.startswith('application/json'):
            response = {
                'learner': profile,
                'context': context,
                'recommendations': recommendations.to_dict('records'),
                'evaluation': evaluation,
                'anon_id': anon_id
            }
            return jsonify(response)
        else:
            return render_template('index.html',
                                  learner=profile,
                                  context=context,
                                  recommendations=recommendations.to_dict('records'),
                                  evaluation=evaluation,
                                  anon_id=anon_id,
                                  courses=recommender.courses.sample(2).to_dict('records'))

    @app.route('/rate', methods=['POST'])
    def rate_course():
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

        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO interactions VALUES (?, ?, ?, ?, ?)',
                  (learner_id, course_id, rating, 10, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        c.execute('INSERT INTO feedback VALUES (?, ?, ?, ?)',
                  (learner_id, course_id, rating, comment))
        c.execute('SELECT AVG(rating) FROM feedback WHERE course_id = ?', (course_id,))
        avg_rating = c.fetchone()[0] or rating
        c.execute('UPDATE interactions SET rating = ? WHERE course_id = ?', (avg_rating, course_id))
        conn.commit()
        conn.close()

        recommender.user_interactions[learner_id][course_id] = rating
        return jsonify({'status': 'success'})

    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        metrics = recommender.get_overall_metrics()
        return jsonify(metrics) if metrics else jsonify({'message': 'No evaluation data'}), 404
