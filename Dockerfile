FROM python:3

ADD app.py /
ADD save.py /
ADD spam.csv /

RUN pip install scikit-learn flask pandas flask-RESTFUL gunicorn
RUN python ./save.py
CMD [ "gunicorn", "-w", "1",  "-b", "0.0.0.0:8000", "app:app" ]