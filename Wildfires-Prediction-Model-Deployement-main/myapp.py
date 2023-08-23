from logging import debug
from flask import Flask , render_template , request
import numpy as np
import model

app = Flask (__name__)

@app.route("/")

def main():
    return render_template("mainpage.html")

@app.route('/predict' , methods=["POST"])
def home():
    data1 =request.form["NDVI"]
    data2 =request.form["LST"]
    data3 =request.form["Burned Area"]
    inputt=np.array([[float(data1),float(data2),float(data3)]])
    pred = model.predictions(inputt)
    print(pred)
   
    return render_template('After.html', data = pred)
   



if __name__ == "__main__" :
    app.run(debug=True)
