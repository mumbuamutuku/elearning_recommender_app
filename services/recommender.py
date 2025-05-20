"""
Recommendation engine implementation
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from models.database import get_all_courses, get_all_interactions
from models.schemas import Recommendation
from services.user_service import get_learner_profile, get_context, has_sufficient_history
from services.ontology_service import get_related_topics
from services.prerequisite_service import get_prerequisite_id


def collaborative_filtering(learner_id):
    """
    Generate recommendations using collaborative filtering
    
    Args:
        learner_id: The ID of the learner
        
    Returns:
        np.array: Scores for each course based on collaborative filtering
    """
    # Get all interaction data
    interactions = get_all_interactions()
    
    # Extract unique IDs
    learner_ids = list(set([x[0] for x in interactions]))
    course_ids = list(set([x[1] for x in interactions]))
    
    # No interactions found
    if not learner_ids or not course_ids:
        return np.zeros(len(get_all_courses()))
    
    # Check if learner exists in interactions
    if learner_id not in learner_ids:
        return np.zeros(len(course_ids))
    
    # Build rating matrix
    rating_matrix = np.zeros((len(learner_ids), len(course_ids)))
    for lid, cid, rating in interactions:
        try:
            rating_matrix[learner_ids.index(lid), course_ids.index(cid)] = rating
        except ValueError:
            # Skip if IDs aren't in our lists (shouldn't happen, but just in case)
            continue
    
    # Calculate similarity between users
    similarity = cosine_similarity(rating_matrix)
    
    # Find our learner's index
    learner_idx = learner_ids.index(learner_id)
    
    # Calculate weighted ratings (avoiding division by zero)
    collab_scores = similarity[learner_idx] @ rating_matrix / (np.sum(similarity[learner_idx]) + 1e-9)
    
    return collab_scores

def content_based_filtering(learner_id):
    """
    Generate recommendations using content-based filtering
    
    Args:
        learner_id: The ID of the learner
        
    Returns:
        np.array: Scores for each course based on content filtering
    """
    # Get all courses
    courses = get_all_courses()
    
    # Extract keywords for TF-IDF
    course_texts = [course[2] for course in courses]  # Keywords
    
    # Create TF-IDF vectors for courses
    vectorizer = TfidfVectorizer()
    course_vectors = vectorizer.fit_transform(course_texts).toarray()
    
    # Get learner's profile and interactions
    learner_profile = get_learner_profile(learner_id)
    
    # If no interactions, use goals as a proxy
    if not learner_profile.interactions:
        # Generate a pseudo-vector based on learner's goals
        learner_text = learner_profile.goals
        learner_vector = vectorizer.transform([learner_text]).toarray()[0]
    else:
        # Create a learner vector from weighted courses they've interacted with
        learner_vector = np.zeros(course_vectors.shape[1])
        for cid, rating in learner_profile.interactions:
            try:
                course_idx = [c[0] for c in courses].index(cid)
                learner_vector += course_vectors[course_idx] * rating
            except ValueError:
                # Skip if course ID not found
                continue
    
    # Calculate similarity between learner vector and course vectors
    content_scores = cosine_similarity([learner_vector], course_vectors)[0]
    
    return content_scores

def get_recommendations(learner_id):
    """
    Generate hybrid recommendations for a learner
    
    Args:
        learner_id: The ID of the learner
        
    Returns:
        list: List of Recommendation objects
    """
    # Get all courses
    courses = get_all_courses()
    
    # Determine weighting based on user history
    if has_sufficient_history(learner_id):
        collab_weight = config.COLLAB_WEIGHT
        content_weight = config.CONTENT_WEIGHT
    else:
        # For new users, rely more on content-based filtering
        collab_weight = 0.2
        content_weight = 0.8
    
    # Get scores from both methods
    collab_scores = collaborative_filtering(learner_id)
    content_scores = content_based_filtering(learner_id)
    
    # Adjust dimensions if necessary
    if len(collab_scores) < len(courses):
        temp = np.zeros(len(courses))
        temp[:len(collab_scores)] = collab_scores
        collab_scores = temp
    
    # Combine scores using weighted approach
    hybrid_scores = collab_weight * collab_scores + content_weight * content_scores
    
    # Create recommendation objects
    # recommendations = [
    #     Recommendation(courses[i][1], hybrid_scores[i], courses[i][0]) 
    #     for i in range(len(courses))
    # ]
    learner_goal = get_learner_profile(learner_id).goals
    related_topics = get_related_topics(learner_goal)

    # Optional: flag for explanation
    boosted_indices = set()

    for i, course in enumerate(courses):
        course_keywords = course[2].lower()  # e.g., "python, visualization"
        if any(topic.lower() in course_keywords for topic in related_topics):
            hybrid_scores[i] *= 1.2
            boosted_indices.add(i)
    
    # Use top N content-based scores to infer learner goals
    top_goal_course_ids = sorted(
        range(len(content_scores)),
        key=lambda i: content_scores[i],
        reverse=True
    )[:3]  # Top 3 as target goals

    goal_course_ids = [courses[i][0] for i in top_goal_course_ids]

    # Get prerequisite course IDs
    prerequisite_ids = get_prerequisite_id(goal_course_ids)

    boosted_indices = set()
    for i, course in enumerate(courses):
        if course[0] in prerequisite_ids:
            hybrid_scores[i] *= 1.2  # Boost prerequisites
            boosted_indices.add(i)


    recommendations = []
    for i in range(len(courses)):
        if i in boosted_indices:
            explanation = f"Recommended based on your goal '{learner_goal}' and related prerequisite topics from our knowledge graph."
        elif content_weight > collab_weight:
            explanation = "Recommended based on your learning goals and course content similarity."
        elif collab_weight > content_weight:
            explanation = "Recommended based on preferences of learners with similar interests."
        else:
            explanation = "Recommended using a balanced mix of your goals and peer ratings."

        recommendations.append(
            Recommendation(
                title=courses[i][1],
                score=hybrid_scores[i],
                course_id=courses[i][0],
                explanation=explanation
            )
        )

    
    # Sort by score
    recommendations.sort(key=lambda x: x.score, reverse=True)
    
    # Apply contextual adjustments
    recommendations = apply_contextual_adjustments(learner_id, recommendations, courses)
    
    # Return top N recommendations
    return recommendations[:config.RECOMMENDATIONS_COUNT]

def apply_contextual_adjustments(learner_id, recommendations, courses):
    """
    Adjust recommendation scores based on context
    
    Args:
        learner_id: The ID of the learner
        recommendations: List of initial Recommendation objects
        courses: List of all courses
        
    Returns:
        list: Adjusted list of Recommendation objects
    """
    context = get_context(learner_id)
    
    # Apply context-specific adjustments
    if context.device == 'mobile' and context.time_of_day == 'night':
        # Boost video content on mobile at night
        course_formats = {course[0]: course[3] for course in courses}
        
        for rec in recommendations:
            if rec.course_id in course_formats and 'video' in course_formats[rec.course_id]:
                rec.score *= config.MOBILE_NIGHT_VIDEO_BOOST
    
    # Re-sort after adjustments
    recommendations.sort(key=lambda x: x.score, reverse=True)
    
    return recommendations