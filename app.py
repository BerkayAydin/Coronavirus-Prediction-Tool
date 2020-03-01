from flask import Flask, render_template
import predict
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/refresh_predictions', methods=['GET'])
def refresh_predictions():
    thread = threading.Thread(target=predict.refresh)
    thread.start()
    return "Successfully refreshed data and predictions."


if __name__ == '__main__':
    app.run()
