# Homework 5 - Q3
# cargar el modelo con pickle y predecir para un lead

import pickle

with open('pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# el pipeline ya trae el DictVectorizer, asi que recibe la lista de dicts directo
prob = pipeline.predict_proba([lead])[0, 1]
print('probabilidad de conversion:', round(prob, 3))
