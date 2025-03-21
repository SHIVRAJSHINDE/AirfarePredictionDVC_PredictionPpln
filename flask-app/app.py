import mlflow
import mlflow.pyfunc

from flask import Flask, request, render_template
from flask_cors import cross_origin


from src.Prediction.predictionFile import ReceiveData
import dagshub

app = Flask(__name__)

mlflow.set_tracking_uri('https://dagshub.com/SHIVRAJSHINDE/AirfarePredictionDVC_PredictionPpln.mlflow')
dagshub.init(repo_owner='SHIVRAJSHINDE',repo_name='AirfarePredictionDVC_PredictionPpln',mlflow=True)

# tracking_uri = "http://localhost:5000"
# mlflow.set_tracking_uri(tracking_uri)


logged_model = 'runs:/b86d75382ed74105b6546b9c899fdc44/Lasso_model'

try:
    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)



    print(model)
except Exception as e:
    print("Error loading model:", e)

@app.route("/")
@cross_origin()

def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()

def predict():
    if request.method=="GET":
        return render_template(home.html)
    
    elif request.method=="POST":

        receiveData_Obj =  ReceiveData()
        
        df = receiveData_Obj.receive_data_from_ui_create_df(Airline = request.form.get('Airline'),
                                        Date_of_Journey = request.form.get('Date_of_Journey'),
                                        Source = request.form.get('Source'),
                                        Destination = request.form.get('Destination'),
                                        Dep_Time = request.form.get('Dep_Time'),
                                        Arrival_Time = request.form.get('Arrival_Time'),
                                        Duration = request.form.get('Duration'),
                                        Total_Stops = request.form.get('Total_Stops'))
        print(df)

        value = receiveData_Obj.execute_pipeline(df)
        prediction_value = model.predict(value)
        print("----------------------------------------------------")
        print(value)
        print("----------------------------------------------------")
        return render_template("prediction.html", prediction=prediction_value)

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
