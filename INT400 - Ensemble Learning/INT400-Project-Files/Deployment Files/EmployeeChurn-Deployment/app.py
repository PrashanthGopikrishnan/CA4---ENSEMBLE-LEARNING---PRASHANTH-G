import numpy as np
from flask import Flask, request, render_template

import joblib


app = Flask(__name__)

# Load saved models
dt_model = joblib.load('models/dt.sav')



# Dictionary of all loaded models
loaded_models = {
    'dt': dt_model,
   

}

# Function to decode predictions 
def decode(pred):
    if pred == 1: return 'Customer Exits'
    else: return 'Customer Stays'

@app.route('/')
def home():
    # Initial rendering
    result = [{'model':'ExtraTree Classifier', 'prediction':' '},
              ]
    
    # Create main dictionary
    maind = {}
    maind['customer'] = {}
    maind['predictions'] = result

    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():

    # List values received from index
    values = [x for x in request.form.values()]

    # new_array - input to models
    new_array = np.array(values).reshape(1, -1)
    print(new_array)
    print(values)
    
    # Key names for customer dictionary custd
    cols = ['DailyRate','EducationField','OverTime','Department','JobRole','MaritalStatus',
            'BusinessTravel','JobInvolvement','JobSatisfaction','EnvironmentSatisfaction',
            'RelationshipSatisfaction','PerformanceRating','JobLevel','StockOptionLevel',
            'Education','WorkLifeBalance','Gender','Age','YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion',
                          'YearsWithCurrManager','HourlyRate','MonthlyRate',
                          'MonthlyIncome','PercentSalaryHike','TotalWorkingYears', 
                          'TrainingTimesLastYear','NumCompaniesWorked','DistanceFromHome' ]

    # Create customer dictionary
    custd = {}
    for k, v in  zip(cols, values):
        custd[k] = v

 

    # Loop through 'loaded_models' dictionary and
    # save predictiond to the list
    predl = []
    for m in loaded_models.values():
        predl.append(decode(m.predict(new_array)[0]))

    result = [
            {'model':'ExtraTree Classifier', 'prediction':predl[0]},
            ]

    # Create main dictionary
    maind = {}
    maind['customer'] = custd
    maind['predictions'] = result

    return render_template('index.html', maind=maind)


if __name__ == "__main__":
    app.run(debug=True)