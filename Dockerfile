#
#
#
FROM python:3.11-slim

#
#
#
WORKDIR /app
RUN pip install torch fastapi transformers uvicorn
COPY server.py app.html ./
EXPOSE 8080

#
#
#
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]