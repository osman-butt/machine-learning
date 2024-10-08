from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import load
app = Flask(__name__)

@app.route('/',methods=['post','get']) # will use get for the first page-load, post for the form-submit
def predict(): # this function can have any name
  try:
    model = load_model('model/leaveStayModel.keras') # the mymodel.keras file was created in Colab, downloaded and uploaded using Filezilla
    credit_score = request.form.get('CreditScore')
    age = request.form.get('Age')
    tenure = request.form.get('Tenure')
    balance = request.form.get('Balance')
    num_of_products = request.form.get('NumOfProducts')
    has_cr_card = request.form.get('HasCrCard')
    is_active_member = request.form.get('IsActiveMember')
    estimated_salary = request.form.get('EstimatedSalary')
    geography = request.form.get('Geography')  # Only one selected: France, Germany, Spain
    gender = request.form.get('Gender')        # Only one selected: Male, Female

    # Check if any required input is missing
    if not all([credit_score, age, tenure, balance, num_of_products, has_cr_card,
                is_active_member, estimated_salary, geography, gender]):
        return render_template('index.html', result='No input(s)')
    
    # Convert the categorical data to numerical data
    geography_france = 1 if geography == 'France' else 0
    geography_germany = 1 if geography == 'Germany' else 0
    geography_spain = 1 if geography == 'Spain' else 0

    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0

    # Prepare the input array
    arr = np.array([[float(credit_score), float(age), float(tenure), float(balance),
                    float(num_of_products), float(has_cr_card), float(is_active_member),
                    float(estimated_salary), float(geography_france), float(geography_germany),
                    float(geography_spain), float(gender_female), float(gender_male)]])
    
    # Load the saved scaler
    scaler = load('model/scaler.bin')
    # Use the saved scale to scale the new customer data
    newCustomer = scaler.transform(arr)
    # Make the prediction
    predictions = model.predict(newCustomer)
    # Return the prediction
    return render_template('index.html', result=str(predictions[0][0]))
        # the result is set, by asking for row=0, column=0. Then cast to string.
  except Exception as e:
    return render_template('index.html', result='error ' + str(e))

if __name__ == '__main__':
	app.run(host='0.0.0.0')
