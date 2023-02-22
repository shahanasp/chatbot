import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from translate import Translator
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    global result
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
    except:
        result ="I cannot understand this statement.Perhaps rephrase it or type it differently?"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

from flask import *
# import pymysql
# con=pymysql.connect(host='localhost',user='root',password='muhammad@123',port=3306,db='sample1')
# cmd=con.cursor()

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = "super secret key"

@app.route('/insert',methods=["post"])
def mx():
    mail=request.form["email"]
    fb=request.form["feedback"]
    cmd.execute("INSERT INTO feedback_table(Mailid,Feedback)  values('"+mail+"','"+fb+"')")
    con.commit()
    return render_template('index.html')

@app.route("/")
def home():
    return render_template("collegewebsite.html")
@app.route("/bot")
def home1():
    return render_template("index.html")
@app.route('/ln',methods=["post"])
def language():
    a=request.form.get("ln1")
    b=request.form.get("ln2")
    session['a']=a
    session['b']=b
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    print(session['b'])
    if session['b']=='ml':
        userText = request.args.get('msg')
        translator = Translator(from_lang="ml", to_lang="en")
        userText = translator.translate(userText)
        res = chatbot_response(userText)
        translator = Translator(from_lang="en", to_lang="ml")
        res = translator.translate(res)
        return res
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.debug = True
    app.run()