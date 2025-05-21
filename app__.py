from flask import Flask
from web.routes import init_routes
from models.database import init_db, populate_sample_data

app = Flask(__name__)

# Initialize database and routes
init_db()
populate_sample_data()
init_routes(app)

if __name__ == '__main__':
    app.run(debug=True)