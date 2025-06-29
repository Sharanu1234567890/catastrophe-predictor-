import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# === Load and preprocess data ===
df = pd.read_csv("natural_disasters_2024.csv")
df['Magnitude'] = df['Magnitude'].fillna(df['Magnitude'].mean())
df['Economic_Loss($)'] = df['Economic_Loss($)'].replace(r'[\$,]', '', regex=True).astype(float)

# Label encode Disaster_Type
le_disaster = LabelEncoder()
df['Disaster_Type'] = le_disaster.fit_transform(df['Disaster_Type'])

# Normalize for Risk Score calculation
df['Fatalities_norm'] = df['Fatalities'] / df['Fatalities'].max()
df['Economic_Loss_norm'] = df['Economic_Loss($)'] / df['Economic_Loss($)'].max()

# Risk Score and Category
df['Risk_Score'] = 0.7 * df['Fatalities_norm'] + 0.5 * df['Economic_Loss_norm']

def categorize_risk(score):
    if score > 0.75:
        return 'High Risk'
    elif score > 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Category'] = df['Risk_Score'].apply(categorize_risk)

# Store max values for normalization during prediction
max_fatalities = df['Fatalities'].max()
max_loss = df['Economic_Loss($)'].max()

# === Prepare Models ===
features = ['Disaster_Type', 'Magnitude']

X_fatalities = df[features + ['Economic_Loss($)']]
y_fatalities = df['Fatalities']
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fatalities, y_fatalities, test_size=0.2, random_state=42)
model_fatalities = RandomForestRegressor(random_state=42)
model_fatalities.fit(X_train_f, y_train_f)

X_loss = df[features + ['Fatalities']]
y_loss = df['Economic_Loss($)']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_loss, y_loss, test_size=0.2, random_state=42)
model_loss = RandomForestRegressor(random_state=42)
model_loss.fit(X_train_l, y_train_l)

# === GUI ===
root = tk.Tk()
root.title("🌍 Catastrophe Impact Prediction Dashboard")
root.geometry("720x700")
root.configure(bg="#e9f0f7")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Segoe UI", 10), background="#e9f0f7")
style.configure("TButton", font=("Segoe UI", 10, "bold"))
style.configure("TEntry", padding=5)

prediction_type = tk.StringVar(value="fatalities")

# === Layout ===
title_frame = tk.Frame(root, bg="#003f5c")
title_frame.pack(fill='x')
title_label = tk.Label(title_frame, text="National Catastrophe Impact Predictor", fg="white", bg="#003f5c", font=("Segoe UI", 18, "bold"))
title_label.pack(pady=15)

frame = tk.Frame(root, bg="#e9f0f7", padx=20, pady=20)
frame.pack(fill='both', expand=True)

result_frame = tk.LabelFrame(root, text="🧮 Prediction Result", bg="#ffffff", font=("Segoe UI", 12, "bold"))
result_frame.pack(fill='x', padx=20, pady=10)

result_text = tk.Text(result_frame, height=6, wrap='word', font=("Segoe UI", 10))
result_text.pack(padx=10, pady=10, fill='x')

footer = tk.Label(root, text="© 2024 Department of AI & ML - National Hackathon", bg="#e9f0f7", font=("Segoe UI", 9))
footer.pack(side="bottom", pady=10)

def toggle_inputs():
    if prediction_type.get() == "fatalities":
        loss_entry.configure(state='normal')
        fatalities_entry.configure(state='disabled')
    else:
        loss_entry.configure(state='disabled')
        fatalities_entry.configure(state='normal')

def predict():
    try:
        disaster = disaster_cb.get()
        magnitude = float(magnitude_entry.get())
        d_type = le_disaster.transform([disaster])[0]

        if prediction_type.get() == "fatalities":
            economic_loss = float(loss_entry.get())
            X_input = np.array([[d_type, magnitude, economic_loss]])
            pred = model_fatalities.predict(X_input)[0]
            fatalities_norm = pred / max_fatalities
            loss_norm = economic_loss / max_loss
        else:
            fatalities = float(fatalities_entry.get())
            X_input = np.array([[d_type, magnitude, fatalities]])
            pred = model_loss.predict(X_input)[0]
            fatalities_norm = fatalities / max_fatalities
            loss_norm = pred / max_loss

        risk_score = 0.7 * fatalities_norm + 0.5 * loss_norm
        risk_category = categorize_risk(risk_score)

        result_text.delete("1.0", tk.END)
        if prediction_type.get() == "fatalities":
            result_text.insert(tk.END, f"🔢 Predicted Fatalities: {pred:.2f} people\n")
        else:
            result_text.insert(tk.END, f"💰 Predicted Economic Loss: ${pred:,.2f}\n")
        result_text.insert(tk.END, f"📊 Risk Score: {risk_score:.2f}\n")
        result_text.insert(tk.END, f"⚠️ Risk Category: {risk_category}\n")

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{str(e)}")

# === Widgets ===
tk.Label(frame, text="Select Prediction Type:", font=("Segoe UI", 10, "bold"), bg="#e9f0f7").grid(row=0, column=0, sticky='w', pady=5)
tk.Radiobutton(frame, text="Predict Fatalities", variable=prediction_type, value="fatalities", command=toggle_inputs, bg="#e9f0f7").grid(row=0, column=1, sticky='w')
tk.Radiobutton(frame, text="Predict Economic Loss", variable=prediction_type, value="loss", command=toggle_inputs, bg="#e9f0f7").grid(row=0, column=2, sticky='w')

labels = ["Disaster Type:", "Magnitude:", "Economic Loss ($):", "Fatalities:"]

for i, text in enumerate(labels):
    tk.Label(frame, text=text, font=("Segoe UI", 10), bg="#e9f0f7").grid(row=i+1, column=0, sticky='w', pady=5)

# Entries and combobox
disaster_cb = ttk.Combobox(frame, values=list(le_disaster.classes_), width=30)
disaster_cb.grid(row=1, column=1, columnspan=2, sticky='w')

magnitude_entry = ttk.Entry(frame, width=33)
magnitude_entry.grid(row=2, column=1, columnspan=2, sticky='w')

loss_entry = ttk.Entry(frame, width=33)
loss_entry.grid(row=3, column=1, columnspan=2, sticky='w')

fatalities_entry = ttk.Entry(frame, width=33)
fatalities_entry.grid(row=4, column=1, columnspan=2, sticky='w')

# Predict button
predict_btn = ttk.Button(frame, text="🔍 Predict Impact", command=predict)
predict_btn.grid(row=5, column=0, columnspan=3, pady=20)

toggle_inputs()
root.mainloop()
