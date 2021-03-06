FROM python:3.7

COPY ./requirements.txt ./requirements.txt
# Install dependencies
RUN pip install -r requirements.txt

WORKDIR .
ADD . .


# Expose port 
EXPOSE 5000

# Run the application:
CMD ["flask", "run", "--host", "0.0.0.0"]