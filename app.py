from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from scipy.stats import mode
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('DPBOS.html')

@app.route('/result', methods =["GET", "POST"])
def gfg():
    DATA_PATH = "Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis = 1)
 
    disease_counts = data["prognosis"].value_counts()
    temp_df = pd.DataFrame({
        "Disease": disease_counts.index,
        "Counts": disease_counts.values
    })
 
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])


    X = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test =train_test_split(
    X, y, test_size = 0.2, random_state = 24)

    def cv_scoring(estimator, X, y):
        return accuracy_score(y, estimator.predict(X))
 
    models = {
        "SVC":SVC(),
        "Gaussian NB":GaussianNB(),
        "Random Forest":RandomForestClassifier(random_state=18)
    }
 
    for model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv = 10,n_jobs = -1,scoring = cv_scoring)
    
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)
 
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds = nb_model.predict(X_test)

    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)

    cf_matrix = confusion_matrix(y_test, preds)


    symptoms = X.columns.values

    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
	    "symptom_index":symptom_index,
	    "predictions_classes":encoder.classes_
    }

    def predictDisease(symptoms):
        symptoms = symptoms.split(",")
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
		
        input_data = np.array(input_data).reshape(1,-1)
	
        rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]
	
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
        predictions = {
		    "rf_model_prediction": rf_prediction,
		    "naive_bayes_prediction": nb_prediction,
		    "svm_model_prediction": nb_prediction,
		    "final_prediction":final_prediction
	    }
        return final_prediction
    if request.method == "POST":
        symptom_name = request.form.get("fname")
        res=predictDisease(symptom_name)
    return '<h1>You have {}</h1>'.format(res)
