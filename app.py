from flask import Flask, render_template
import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/refresh_predictions', methods=['GET'])
def refresh_predictions():
    predict.refresh()
    return "Successfully refreshed data and predictions."


if __name__ == '__main__':
    app.run()
