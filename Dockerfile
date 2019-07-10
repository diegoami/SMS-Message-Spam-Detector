FROM python:3
RUN apt update
RUN apt-get -y install default-jdk
RUN pip install scikit-learn flask pandas flask-RESTFUL gunicorn sklearn2pmml
RUN mkdir /opt
WORKDIR /opt
RUN mkdir data

ADD spam.csv .
ADD app.py .
ADD save.py .
ADD pmml.py .
ADD cmds.sh .

CMD [ "/bin/sh", "./cmds.sh" ]