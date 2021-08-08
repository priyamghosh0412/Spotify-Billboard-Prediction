from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Boosted_Model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data7 = float(request.form['g'])
    data8 = float(request.form['h'])
    data9 = float(request.form['i'])
    data10 = int(request.form['j'])
    arr = np.array([[data1, data2, data3, data4, data5, data7, data8, data9, data10]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)


  
       