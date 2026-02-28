FROM python:3.13-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv \
    && PIPENV_VENV_IN_PROJECT=0 pipenv install --system --ignore-pipfile \
    && pip install --no-cache-dir psycopg2-binary

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
