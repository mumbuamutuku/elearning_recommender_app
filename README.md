# Learning Recommendation System

A hybrid recommendation system for online courses that combines content-based and collaborative filtering techniques with contextual personalization.

## Features

- **Hybrid Recommendations**: Combines content-based and collaborative filtering approaches
- **Personalization**: Considers learner profiles, goals, and preferences
- **Context Awareness**: Adapts recommendations based on device type and time of day
- **Evaluation Metrics**: Tracks precision, recall, diversity, novelty, and coverage
- **Explainable AI**: Provides transparent explanations for recommendations
- **Feedback Integration**: Incorporates user ratings to improve future recommendations

## Technologies

- Python 3
- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data processing)
- SQLite (database)
- TF-IDF (text vectorization)
- Cosine Similarity (content matching)

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/learning-recommender.git
   cd learning-recommender

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies:
```bash
    pip install -r requirements.txt

## Data Preparation
Place your course data in CSV format in the data/ directory. The system expects columns including:

 - course_name
 - course_description
 - skills
 - difficulty_level
 - course_rating

## Usage
1. Run the application:
```bash
python app.py
Access the web interface at http://localhost:5000

2. API Endpoints
- GET / - Homepage with sample courses

- POST /recommend - Get recommendations (requires query)

- POST /rate - Submit a course rating

- GET /metrics - View system evaluation metrics

- GET /user/recommendations - Personalized recommendations interface

- POST /feedback - Submit detailed course feedback

## Configuration
The system can be configured by modifying:

- models/database.py - Database schema and sample data

- services/ - Core recommendation logic and evaluation

- templates/ - Web interface templates

## Evaluation Metrics
The system tracks several recommendation quality metrics:

- Precision: Percentage of recommended items that are relevant

- Recall: Percentage of relevant items that are recommended

- Diversity: Variety among recommended items

- Novelty: How unexpected recommendations are

- Coverage: Percentage of catalog items recommended