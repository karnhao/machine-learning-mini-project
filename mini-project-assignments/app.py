import pandas as pd
import gradio as gr
import joblib
# "Artifacts"
pipeline = joblib.load("pipeline.joblib")
label_pipeline = joblib.load("label_pipeline.joblib")

ownerships = joblib.load("ownerships.joblib")
transactions = joblib.load("transactions.joblib")
locations = joblib.load("locations.joblib")
furnishings = joblib.load("furnishings.joblib")
open_covereds = joblib.load("open_covereds.joblib")
overlookings = joblib.load("overlooking.joblib")

def predict(location, carpet_area, transaction, furnishing, overlooking, bathroom, balcony, ownership , open_covered, car_parking):

    sample = dict()
    sample["location"] = location
    sample["Carpet Area"] = carpet_area
    sample["Transaction"] = transaction
    sample["Furnishing"] = furnishing
    sample["overlooking"] = "/".join(overlooking)
    sample["Bathroom"] = bathroom
    sample["Balcony"] = balcony
    sample["Ownership"] = ownership
    sample["open_covered"] = open_covered
    sample["car_parking_count"] = car_parking

    price = pipeline.predict(pd.DataFrame([sample]))
    price = label_pipeline.inverse_transform([price])
    
    return int(price[0][0])

# https://www.gradio.app/guides
with gr.Blocks() as blocks:
   
    location = gr.Dropdown(locations, value=locations[0], label="location")
    carpet_area = gr.Number(label="Carpet Area", value=1, minimum=1)
    transaction = gr.Radio(transactions, label="Transaction")            
    furnishing = gr.Radio(furnishings, label="Furnishing") 
    overlooking = gr.CheckboxGroup(overlookings, label="overlooking")            
    bathroom = gr.Slider(label="Bathroom", minimum=0, maximum=10, step=1,info = "How many bathrooms are there?")
    balcony = gr.Slider(label="Balcony", minimum=0, maximum=10, step=1,info = "How many balcony are there?")
    ownership = gr.Radio(ownerships, label="Ownership") 
    open_covered = gr.Radio(open_covereds, label="Car parking is open or covered?") 
    car_parking = gr.Slider(label="Car parking count", minimum=0, maximum=10, step=1,info = "How many car that can park there?")



    price = gr.Number(label="Price")

    inputs = [location, carpet_area, transaction, furnishing, overlooking, bathroom, balcony, ownership , open_covered, car_parking]
    outputs = [price]

    predict_btn = gr.Button("Predict")
    predict_btn.click(predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    blocks.launch() # Local machine only
    # blocks.launch(server_name="0.0.0.0") # LAN access to local machine
    # blocks.launch(share=True) # Public access to local machine
