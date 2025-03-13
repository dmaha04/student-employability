import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split

# Load or train models
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open("logistic_regression.pkl", "rb") as model_file:
        logistic_regression = pickle.load(model_file)

    with open("perceptron.pkl", "rb") as model_file:
        perceptron = pickle.load(model_file)

except (FileNotFoundError, pickle.UnpicklingError):
    print("Training models...")

    df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")
    X = df.iloc[:, 1:-2].values
    y = (df["CLASS"] == "Employable").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train_scaled, y_train)

    perceptron = Perceptron(random_state=42)
    perceptron.fit(X_train_scaled, y_train)

    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("logistic_regression.pkl", "wb") as model_file:
        pickle.dump(logistic_regression, model_file)
    with open("perceptron.pkl", "wb") as model_file:
        pickle.dump(perceptron, model_file)


# Prediction function
def predict_employability(name, ga, mos, pc, ma, sc, api, cs, model_choice):
    name = name.strip() if name else "The candidate"

    input_data = np.array([[ga, mos, pc, ma, sc, api, cs]])
    input_scaled = scaler.transform(input_data)

    model_map = {
        "Logistic Regression": logistic_regression,
        "Perceptron": perceptron
    }

    model = model_map.get(model_choice, logistic_regression)
    prediction = model.predict(input_scaled)

    return f"✅ {name} is Employable!" if prediction[0] == 1 else f"❌ {name} needs to improve!"


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Employability Evaluation 🚀\nAssess a candidate's employability based on key skills.")

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Candidate Name")
            ga = gr.Slider(1, 5, step=1, label="General Appearance")
            mos = gr.Slider(1, 5, step=1, label="Manner of Speaking")
            pc = gr.Slider(1, 5, step=1, label="Physical Condition")
            ma = gr.Slider(1, 5, step=1, label="Mental Alertness")
            sc = gr.Slider(1, 5, step=1, label="Self Confidence")
            api = gr.Slider(1, 5, step=1, label="Ability to Present Ideas")
            cs = gr.Slider(1, 5, step=1, label="Communication Skills")
            model_choice = gr.Radio(["Logistic Regression", "Perceptron"], label="Choose Prediction Model")

            predict_btn = gr.Button("Evaluate Candidate")

        with gr.Column():
            result_output = gr.Textbox(label="Employability Prediction", interactive=False)

    # Button Click Event
    predict_btn.click(
        fn=predict_employability,
        inputs=[name, ga, mos, pc, ma, sc, api, cs, model_choice],
        outputs=[result_output]
    )

# Launch the app
app.launch(share=True)
