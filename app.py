import flask
import pickle

# Use pickle to load in the pre-trained model
with open(f'model/model.pk', 'rb') as f:
    model = pickle.load(f)

with open(f'model/vect.pk', 'rb') as f:
    vectorizer = pickle.load(f)
# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        review = flask.request.form['review']

        # Make DataFrame for model
        # Get the model's prediction
        prediction = model.predict(vectorizer.transform([review]))[0]
        prediction_proba = model.predict_proba(vectorizer.transform([review]))[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Review':review},
                                     result=prediction,
                                     probabilities=prediction_proba,
                                     count=len(prediction_proba)
                                     )

if __name__ == '__main__':
    app.run()