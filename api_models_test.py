import google.generativeai as genai

genai.configure(api_key="AIzaSyBFBeomaTTGWT_8lUeM3MuFEkdOZXM8-yA")

for model in genai.list_models():
    print(model.name)
