# app.py (Updated with Proactive Recs, Reasons, and Dish Matching)

# 1. Imports
from flask import Flask, render_template, request, redirect, url_for, flash, session
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
login_manager.login_view = 'login'

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
    
    ### NEW FEATURE ###
    # Ensure 'Popular Dishes' column exists and is filled with strings
    if 'Popular Dishes' not in df.columns:
        df['Popular Dishes'] = ''
    df['Popular Dishes'] = df['Popular Dishes'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['name_for_matching']).drop_duplicates()
    all_restaurants_list = sorted(df['Name'].unique())
    print("✅ AI Model built successfully.")
except FileNotFoundError:
    df = None
    print("CRITICAL: 'restaurants_cleaned.csv' not found. Recommendations will not work.")

# 7. AI Helper Functions (Updated to return structured data)

### NEW FEATURE ###
def get_recommendations_from_text(user_input):
    user_vec = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-10:][::-1]
    
    recs_df = df.iloc[sim_indices]
    recommendations = recs_df.to_dict('records')
    
    # Add reason and dish matching
    for rec in recommendations:
        rec['reason'] = f"Matches your craving for '{user_input}'"
        # Check if the craving text appears in the popular dishes for that restaurant
        if user_input.lower() in str(rec.get('Popular Dishes', '')).lower():
            rec['matched_dish'] = user_input
        else:
            rec['matched_dish'] = None
            
    return recommendations

### NEW FEATURE ###
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
    
    recs_df = df.iloc[restaurant_indices]
    recommendations = recs_df.to_dict('records')

    # Add reason
    for rec in recommendations:
        rec['reason'] = f"Similar to restaurants you like, such as '{name}'"
        rec['matched_dish'] = None # No dish matching for this type of rec

    return recommendations

# 8. Web Routes (Updated)
@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    recommendations = None
    proactive_recs = None
    error = None

    if df is None:
        error = "Recommendation engine is offline. Please check server data files."
        return render_template('index.html', error=error)

    ### NEW FEATURE: PROACTIVE RECS on GET request ###
    if request.method == 'GET' and 'history' in session and session['history']:
        last_craving = session['history'][-1] # Get the last search
        proactive_recs = get_recommendations_from_text(last_craving)
        # Update reason for proactive recs
        for rec in proactive_recs:
             rec['reason'] = f"Because you recently searched for '{last_craving}'"


    if request.method == 'POST':
        restaurant_name = request.form.get('restaurant_name')
        taste_query = request.form.get('taste_query')

        if taste_query:
            recommendations = get_recommendations_from_text(taste_query)
            session.setdefault('history', []).append(taste_query) # Add to history
            session.modified = True
        elif restaurant_name:
            recommendations = get_recommendations(restaurant_name)
            # We can also add liked restaurants to history for more complex logic later
            session.setdefault('history', []).append(restaurant_name)
            session.modified = True
            
        if not recommendations:
            error = "Could not find any recommendations. Please try a different query."

    return render_template(
        'index.html',
        recommendations=recommendations,
        proactive_recs=proactive_recs,
        all_restaurants=all_restaurants_list,
        error=error
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
        
        ### NEW FEATURE ###
        # Initialize user's history on successful login
        session['history'] = []

        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    ### NEW FEATURE ###
    # Clear history on logout
    session.pop('history', None)
    logout_user()
    return redirect(url_for('login'))

# 9. Main Execution Block
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
