# app.py (Complete and Final Version)

# 1. Imports
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 2. App Initialization and Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-secret-key-you-should-change'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Avengers/2005@localhost/food_recommender'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 3. Extensions Initialization
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # This is crucial for @login_required

# 4. Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# 5. User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 6. AI Model Loading
try:
    df = pd.read_csv('restaurants_cleaned.csv')
    df['name_for_matching'] = df['Name'].str.lower().str.strip()
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['name_for_matching']).drop_duplicates()
    all_restaurants_list = sorted(df['Name'].unique())
    print("✅ AI Model built successfully.")
except FileNotFoundError:
    df = None
    print("CRITICAL: 'restaurants_cleaned.csv' not found. Recommendations will not work.")

# 7. AI Helper Functions
def get_recommendations_from_text(user_input):
    user_vec = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-10:][::-1]
    return df.iloc[sim_indices]

def get_recommendations(name):
    name_standardized = name.lower().strip()
    if name_standardized not in indices:
        return None
    idx = indices[name_standardized]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    restaurant_indices = [i[0] for i in sim_scores]
    return df.iloc[restaurant_indices]

# 8. Web Routes
@app.route('/', methods=['GET', 'POST'])
@login_required # Protects this page
def home():
    recommendations = None
    error = None
    selected_restaurant = ""

    if df is None:
        error = "Recommendation engine is offline. Please check server data files."
        return render_template('index.html', error=error)

    if request.method == 'POST':
        restaurant_name = request.form.get('restaurant_name')
        taste_query = request.form.get('taste_query')
        selected_restaurant = restaurant_name

        if taste_query:
            recs_df = get_recommendations_from_text(taste_query)
            if not recs_df.empty:
                recommendations = recs_df.to_dict('records')
        elif restaurant_name:
            recs_df = get_recommendations(restaurant_name)
            if recs_df is not None and not recs_df.empty:
                recommendations = recs_df.to_dict('records')

    return render_template(
        'index.html',
        recommendations=recommendations,
        all_restaurants=all_restaurants_list,
        error=error,
        selected_restaurant=selected_restaurant
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))
        login_user(user, remember=True)
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# 9. Main Execution Block
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Creates tables if they don't already exist
    app.run(debug=True)