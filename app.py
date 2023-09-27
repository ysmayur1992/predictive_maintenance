from flask import Flask,request,render_template


from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

## Route for a home page

@app.route('/')
def index():
    return render_template('homepage.html') 

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('input.html')
    else:
        data=CustomData(
            type=request.form.get('type'),
            air_temperature=float(request.form.get('air_temperature')),
            process_temperature=float(request.form.get('process_temperature')),
            rotational_speed=float(request.form.get('rotational_speed')),
            torque=float(request.form.get('torque')),
            tool_wear=float(request.form.get('tool_wear')),
            power=float(request.form.get('power'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        res = "Your Machine is good to go !!!"
        if results[0] == 1:
            res = "The Machine Urgently Needs Maintenance"
        return render_template('input.html',results=res)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        