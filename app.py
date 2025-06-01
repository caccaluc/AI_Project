import gradio as gr
import pandas as pd
import joblib


# Modell laden
model = joblib.load("models/best_model.pkl")

# Spalten (nur Beispiel – anpassen an dein tatsächliches Feature Encoding)
def prepare_input(brand, vehicle_class, fuel_type, engine_size_l, cylinders):
    # Manuell ein DataFrame erzeugen mit denselben Spalten wie im Training (nach Encoding!)
    # Beispielhafte Struktur (genaues Encoding muss evtl. angepasst werden!)
    df = pd.DataFrame({
        "engine_size_l": [engine_size_l],
        "cylinders": [cylinders],
        f"brand_{brand}": [1],
        f"vehicle_class_{vehicle_class}": [1],
        f"fuel_type_{fuel_type}": [1],
    })

    # Fehlende Dummy-Spalten auffüllen mit 0 (wenn nicht im Input enthalten)
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    return df[model.feature_names_in_]

# Vorhersagefunktion
def predict_consumption(brand, vehicle_class, fuel_type, engine_size_l, cylinders):
    input_df = prepare_input(brand.upper(), vehicle_class, fuel_type, engine_size_l, cylinders)
    prediction = model.predict(input_df)[0]
    return f"{prediction:.2f} Liter / 100 km"

# Gradio UI bauen
demo = gr.Interface(
    fn=predict_consumption,
    inputs=[
        gr.Textbox(label="Marke (BRAND, z.B. BMW)"),
        gr.Dropdown(choices=["Car - Small", "Car - Midsize", "Car - Large", "SUV", "Van/Minivan", "Pickup Truck", "Special"], label="Fahrzeugklasse"),
        gr.Dropdown(choices=["gasoline regular", "gasoline premium", "diesel"], label="Kraftstofftyp"),
        gr.Slider(minimum=0.8, maximum=8.4, step=0.1, label="Motorgröße (Liter)"),
        gr.Slider(minimum=2, maximum=10, step=1, label="Zylinderanzahl"),
    ],
    outputs=gr.Textbox(label="Vorhergesagter Verbrauch (L/100 km)"),
    title="Fahrzeugverbrauchs-Prognose",
    description="Diese App sagt den durchschnittlichen Verbrauch eines Fahrzeugs in Litern pro 100 km voraus."
)

if __name__ == "__main__":
    demo.launch()
