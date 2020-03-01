from flask import Flask, render_template
import predict
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictions_confirmed')
def predictions_confirmed():
    return render_template('fig_c.html')


# @app.route('/predictions_deaths')
# def predictions_deaths():
#     return render_template('fig_d.html')


# @app.route('/predictions_recovered')
# def predictions_recovered():
#     return render_template('fig_r.html')


@app.route('/current_confirmed')
def current_confirmed():
    return render_template('curr_c.html')


# @app.route('/current_deaths')
# def current_deaths():
#     return render_template('curr_d.html')


# @app.route('/current_recovered')
# def current_recovered():
#     return render_template('curr_r.html')


@app.route('/refresh_predictions', methods=['GET'])
def refresh_predictions():
    thread1 = threading.Thread(target=predict.refresh, args=('confirmed',))
    thread1.start()
    # thread2 = threading.Thread(target=predict.refresh, args=('deaths',))
    # thread2.start()
    # thread3 = threading.Thread(target=predict.refresh, args=('recovered',))
    # thread3.start()
    thread4 = threading.Thread(target=predict.refresh, args=('curr_confirmed',))
    thread4.start()
    # thread5 = threading.Thread(target=predict.refresh, args=('curr_deaths',))
    # thread5.start()
    # thread6 = threading.Thread(target=predict.refresh, args=('curr_recovered',))
    # thread6.start()
    return "Successfully started refreshing data and predictions."


if __name__ == '__main__':
    app.run()
