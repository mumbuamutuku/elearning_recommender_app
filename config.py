"""
Configuration settings for the learning platform
"""
# Database settings
COURSE_DATA = './data/coursera.csv'

# Recommendation settings
COLLAB_WEIGHT = 0.7
CONTENT_WEIGHT = 0.3
RECOMMENDATIONS_COUNT = 3

# Context adjustment factors
MOBILE_NIGHT_VIDEO_BOOST = 1.2