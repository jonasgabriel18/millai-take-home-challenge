# MillAI Ad Recommendation System
Data Scientist Take-Home Challenge

This project is a recommendation system for advertisements, using a machine learning model trained with XGBoost. The application provides an API in FastAPI and an interactive interface developed with Streamlit.

## Run the code

1. Clone the repository

```bash
git clone https://github.com/seu-usuario/millai-take-home-challenge.git
cd millai-take-home-challenge
```

2. Create an Virtual Env

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install requirements

```bash
pip install -r requirements.txt
```

4. Run FastAPI

```bash
fastapi run ./app.py
```

5. Run streamlit webpage

In another terminal tab, inside the project, run:

```bash
streamlit run web/main.py
```