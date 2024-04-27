from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
# nltk.download("punkt")
from nltk.tokenize import word_tokenize


app = Flask(__name__,static_url_path='/static')

data = pd.read_csv('C:/Users/ramya/Videos/AnxietyDipressionPanic/PinkyProject/panic_disorder_dataset.csv')
panic_y = data["Panic Disorder Diagnosis"]
panic_X = data.drop(columns=["Panic Disorder Diagnosis", "Participant ID"])
df=pd.read_csv("C:/Users/ramya/Videos/AnxietyDipressionPanic/PinkyProject/mental_health.csv",encoding="ISO-8859-1")
msg=df.text
msg=msg.str.replace('[^a-zA-Z0-9]+'," ")
text_corpus = ' '.join(df['text'].values)
wordcloud = WordCloud(width=800, height=400, background_color='pink').generate(text_corpus)
stemmer=PorterStemmer()
msg=msg.apply(lambda line:[stemmer.stem(token.lower()) for token in word_tokenize(line)]).apply(lambda token:" ".join(token))
msg=msg.apply(lambda line:[token for token in word_tokenize(line) if len(token)>2]).apply(lambda y:" ".join(y))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
data_vec=tf.fit_transform(msg)
from joblib import Parallel, delayed 
import joblib 

knn_from_joblib = joblib.load('C:/Users/ramya/Videos/AnxietyDipressionPanic/PinkyProject/dipression.pkl') 

def process_categorical_features(_dataframe):
    features = {
        "Lifestyle Factors": {'Sleep quality': 1, 'Exercise': 2, 'Diet': 3},
        "Social Support": {'High': 3, 'Moderate': 2, 'Low': 1},
        "Coping Mechanisms": {'Socializing': 1, 'Exercise': 2, 'Seeking therapy': 3, 'Meditation': 4},
        "Substance Use": {'None': 0, 'Drugs': 1, 'Alcohol': 2},
        "Psychiatric History": {'Bipolar disorder': 1, 'Anxiety disorder': 2, 'Depressive disorder': 3, 'None': 0},
        "Medical History": {'Diabetes': 1, 'Asthma': 2, 'None': 0, 'Heart disease': 3},
        "Demographics": {'Rural': 1, 'Urban': 0},
        "Impact on Life": {'Mild': 1, 'Significant': 3, 'Moderate': 2},
        "Severity": {'Mild': 1, 'Moderate': 2, 'Severe': 3},
        "Symptoms": {'Shortness of breath': 1, 'Panic attacks': 2, 'Chest pain': 3, 'Dizziness': 4, 'Fear of losing control': 5},
        "Current Stressors": {'Moderate': 2, 'High': 3, 'Low': 1},
        "Personal History": {'Yes': 1, 'No': 0},
        "Family History": {'Yes': 1, 'No': 0},
        "Gender": {'Male': 0, 'Female': 1}
    }
    for key in features:
        _dataframe[key] = _dataframe[key].map(features[key])
        
    return _dataframe

panic_X = process_categorical_features(panic_X)
panic_X.fillna(0,inplace=True)
X_train,X_test,y_train, y_test = train_test_split(panic_X,panic_y ,random_state=104,  test_size=0.25, shuffle=True) 
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
prediction_text = ""
predicted_mood = ""

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/showapp')
def showapp():
    return render_template('app.html')

@app.route('/remainder')
def raminder():
    return render_template('remainder.html')

@app.route('/sleep')
def sleep():
    return render_template('sleep.html')

@app.route('/daily')
def daily():
    return render_template('daily.html')

@app.route('/tic')
def tic():
    return render_template('tic.html')

@app.route('/rock')
def rock():
    return render_template('rock.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/index')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/index')
def index():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        if user:
            return render_template('index.html', user=user)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')


@app.route('/predictPanic',methods=['GET','POST'])
def predictPanic():
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        familyHistory = request.form.get('familyHistory')
        personalHistory = request.form.get('personalHistory')
        currentStressors = request.form.get('currentStressors')
        symptoms = request.form.get('symptoms')
        severity = request.form.get('severity')
        impactOnLife = request.form.get('impactOnLife')
        demographics = request.form.get('demographics')
        medicalHistory = request.form.get('medicalHistory')
        psychiatricHistory = request.form.get('psychiatricHistory')
        substanceUse = request.form.get('substanceUse')
        copingMechanisms = request.form.get('copingMechanisms')
        socialSupport = request.form.get('socialSupport')
        lifestyleFactors = request.form.get('lifeStyleFactors')
        ipdata = {
            'Age': age,
            'Gender': gender,
            'Family History': familyHistory,
            'Personal History': personalHistory,
            'Current Stressors': currentStressors,
            'Symptoms': symptoms,
            'Severity': severity,
            'Impact on Life': impactOnLife,
            'Demographics': demographics,
            'Medical History': medicalHistory,
            'Psychiatric History': psychiatricHistory,
            'Substance Use': substanceUse,
            'Coping Mechanisms': copingMechanisms,
            'Social Support': socialSupport,
            'Lifestyle Factors': lifestyleFactors
        }
        ipdf = pd.DataFrame([ipdata])

        ipdf = process_categorical_features(ipdf)
        ipdf.fillna(0,inplace=True)
        result = dtree.predict(ipdf)
        ipdata['Panic Disorder Diagnosis'] = result[0]
        data.loc[len(data)] = ipdata
        data.to_csv('C:/Users/ramya/Videos/AnxietyDipressionPanic/PinkyProject/panic_disorder_dataset.csv', index=False)
        global prediction_text, predicted_mood
        if result[0]==1:
            prediction_text = "Yes"
            return render_template('app.html',prediction_text=prediction_text,predicted_mood=predicted_mood)
        else:
            prediction_text = "No"
            return render_template('app.html',prediction_text=prediction_text,predicted_mood=predicted_mood)

@app.route('/predictAnxiety',methods=['POST','GET'])
def predictAnxiety():
    if request.method == 'POST':
        mood = request.form.get('mood')
        sv_frompkl = joblib.load('C:/Users/ramya/Videos/AnxietyDipressionPanic/PinkyProject/dipression.pkl') 
        moodPred = sv_frompkl.predict(tf.transform([mood]))
        global prediction_text, predicted_mood
        if moodPred == 1:
            predicted_mood = "Anxiety"
            return render_template('app.html',prediction_text=prediction_text,predicted_mood=predicted_mood)
        else:
            predicted_mood = "Dipression"
            return render_template('app.html',prediction_text=prediction_text,predicted_mood=predicted_mood)

if __name__ == "__main__":
    app.run(debug=True)
    
