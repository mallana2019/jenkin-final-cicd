FROM python:3.7

RUN pip install --upgrade pip

COPY . .
  
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN python train.py

CMD ["python","app.py"]
