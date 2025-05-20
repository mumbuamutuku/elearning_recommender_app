"""
Data models and schemas for the learning platform
"""

class Learner:
    """Learner profile model"""
    def __init__(self, id, name, age, goals, preferences, interactions=None):
        self.id = id
        self.name = name
        self.age = age
        self.goals = goals
        self.preferences = preferences
        self.interactions = interactions or []

    @classmethod
    def from_db_row(cls, row, interactions=None):
        return cls(id=row[0], name=row[1], age=row[2], goals=row[3], preferences=row[4], interactions=interactions)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'goals': self.goals,
            'preferences': self.preferences,
            'interactions': [i.to_dict() for i in self.interactions]
        }

class Course:
    """Course model"""
    def __init__(self, id, title, keywords, format, prerequisites=None):
        self.id = id
        self.title = title
        self.keywords = keywords
        self.format = format
        self.prerequisites = prerequisites or []

    @classmethod
    def from_db_row(cls, row, prerequisites=None):
        return cls(id=row[0], title=row[1], keywords=row[2], format=row[3], prerequisites=prerequisites)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'keywords': self.keywords,
            'format': self.format,
            'prerequisites': self.prerequisites
        }

class Interaction:
    """Interaction between a learner and a course"""
    def __init__(self, learner_id, course_id, rating, time_spent, timestamp):
        self.learner_id = learner_id
        self.course_id = course_id
        self.rating = rating
        self.time_spent = time_spent
        self.timestamp = timestamp

    @classmethod
    def from_db_row(cls, row):
        return cls(*row)

    def to_dict(self):
        return {
            'learner_id': self.learner_id,
            'course_id': self.course_id,
            'rating': self.rating,
            'time_spent': self.time_spent,
            'timestamp': self.timestamp
        }

class Feedback:
    """Feedback model"""
    def __init__(self, learner_id, course_id, rating, comment):
        self.learner_id = learner_id
        self.course_id = course_id
        self.rating = rating
        self.comment = comment

    @classmethod
    def from_db_row(cls, row):
        return cls(*row)

    def to_dict(self):
        return {
            'learner_id': self.learner_id,
            'course_id': self.course_id,
            'rating': self.rating,
            'comment': self.comment
        }

class Prerequisite:
    """Course prerequisite relationship"""
    def __init__(self, course_id, prerequisite_id):
        self.course_id = course_id
        self.prerequisite_id = prerequisite_id

    @classmethod
    def from_db_row(cls, row):
        return cls(*row)

    def to_dict(self):
        return {
            'course_id': self.course_id,
            'prerequisite_id': self.prerequisite_id
        }

class Context:
    """User context model"""
    def __init__(self, time_of_day, device):
        self.time_of_day = time_of_day
        self.device = device

    def to_dict(self):
        return {
            'time': self.time_of_day,
            'device': self.device
        }

class Recommendation:
    """Recommendation model with course title and score"""
    def __init__(self, title, score, course_id=None, explanation=""):
        self.title = title
        self.score = score
        self.course_id = course_id
        self.explanation = explanation

    def to_dict(self):
        return {
            'title': self.title,
            'score': self.score,
            'course_id': self.course_id,
            'explanation': self.explanation
        }

class Evaluation:
    """Evaluation metrics model"""
    def __init__(self, precision, engagement):
        self.precision = precision
        self.engagement = engagement

    def to_dict(self):
        return {
            'precision': self.precision,
            'engagement': self.engagement
        }
