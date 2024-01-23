# !pip install gradio ipywidgets
import pandas as pd
import gradio as gr
import joblib

# "Artifacts"
pipeline = joblib.load("pipeline.joblib")
label_pipeline = joblib.load("label_pipeline.joblib")
cities = joblib.load("cities.joblib")

def predict(city, location, area, bedrooms, baths):
    sample = dict()
    sample["city"] = city
    sample["location"] = location
    sample["Area_in_Marla"] = area # Column names matching feature names
    sample["bedrooms"] = bedrooms
    sample["baths"] = baths

    price = pipeline.predict(pd.DataFrame([sample]))
    price = label_pipeline.inverse_transform([price])
    return int(price[0][0])

# https://www.gradio.app/guides
with gr.Blocks() as blocks:
    city = gr.Dropdown(cities, value=cities[0], label="City")
    location = gr.Textbox(label="Location")
    area = gr.Number(label="Area", value=1, minimum=0.5, step=0.5)
    bedrooms = gr.Slider(label="Bedrooms", minimum=0, maximum=10, step=1)
    baths = gr.Slider(label="Baths", minimum=0, maximum=10, step=1)
    price = gr.Number(label="Price")

    inputs = [city, location, area, bedrooms, baths]
    outputs = [price]

    predict_btn = gr.Button("Predict")
    predict_btn.click(predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    blocks.launch() # Local machine only
    # blocks.launch(server_name="0.0.0.0") # LAN access to local machine
    # blocks.launch(share=True) # Public access to local machine
