# SMS-Message-Spam-Detector

Simple Spam detector, to demonstrate publishing a text classifier that can be accessed as a test service

## SET UP PYTHON ENVIRONMENT

* Install a python local environment and the following libraries: `scikit-learn flask pandas flask-RESTFUL gunicorn sklearn2pmml`
* Execute _save.py_ to create the pickle file for the classifier. The latest output of this execution is in [save_out.txt](save_out.txt)
* Execute _pmml.py_ to export the model to a pmml file
* Execute _app.py_ to start a web service that can be accessed through a web call, or `gunicorn -w 1 -b 0.0.0.0:8000 app:app` to start a gunicorn server on port 8000

## CREATE DOCKER CONTAINER

To create and start a docker container execute the following commands.
The PMML and model files are saved in <YOUR_DATA_DIRECTORY> of the current directory

```
mkdir -p <YOUR_DATA_DIRECTORY>
docker build -t spam_detector . 
docker run  -d -p 8000:8000 -v <YOUR_DATA_DIRECTORY>:/opt/data spam_detector:latest
```



## ACCESS THE MODEL

The model can be accessed with a REST call, that will return 1 in case of Spam, 0 otherwise. Note that we created two models, one with logistic regression, the other one with Naive Bayes. Only the model having logistic regression can be exported to PMML. Their behaviour is different, as can be seen in this example

```
curl   -d "message=Winner" -X POST http://127.0.0.1:8000/predict

{"nm_prediction": 1, "lr_prediction": 0}

curl   -d "message=Congratulations YOU'VE Won. You're a Winner in our August 1000 Prize Draw" -X POST http://127.0.0.1:5000/predict

{"nm_prediction": 1, "lr_prediction": 0}
```

## EXPORT THE PMML MODEL

With _pmml.py_ the model can be saved to a PMML file, that can be used in a JAVA based application. 
See http://github.com/diegoami/DA_spamdetector_scikit_pmml

The logistic regression model delivers the following confusion matrix and precision / recall on the test set of 0.98 / 0.86

|   |   |
|---|---|
|4822|3|
|32|715|
