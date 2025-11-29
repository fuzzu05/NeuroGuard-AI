import os
import traceback
import matplotlib

matplotlib.use("Agg")

import shap
from flask import Flask, render_template, jsonify, request, send_from_directory
from datetime import datetime
import matplotlib.pyplot as plt
from email.message import EmailMessage
import smtplib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from twilio.rest import Client

DOCTOR_EMAIL = os.environ.get("DOCTOR_EMAIL", "shailivish7071@gmail.com")
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "testing6767sigma@gmail.com")
SMTP_PASS = os.environ.get("SMTP_PASS", "zejfjhfxfzxaonyt")
TWILIO_SID = os.environ.get("TWILIO_SID", "ACaa08d08b4956f8cf877e701573937a2e")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH", "7268c72c8d0cbc87e554e09990e17375")
TWILIO_NUMBER = os.environ.get("TWILIO_NUMBER", "+19893732534")
ALERT_NUMBER = os.environ.get("ALERT_NUMBER", "+918591196751")
DOCTOR_PHONE = os.environ.get("DOCTOR_PHONE", None)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", None)

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

DISEASE_LABELS = {
    0: "NEUTRAL",
    1: "SEIZURE",
    2: "ALZHEIMER",
    3: "PARKINSON",
    4: "ADHD",
    5: "STROKE",
    6: "TUMOR",
}

app = Flask(_name_)
eeg_features_df = None
eeg_labels = None
X_train = X_test = y_train = y_test = None
cnn_model = None
normal_indices = anomaly_indices = None
scaler = None
feature_names = None
shap_explainer = None

dataBuffer_server = []
DATA_BUFFER_MAX = 300

try:
    client = Client(TWILIO_SID, TWILIO_AUTH)
except Exception:
    client = None


def generate_insights(pred_value, eeg_row):
    eeg_np = np.array(eeg_row)
    insights = []
    power = np.mean(np.abs(eeg_np))

    if power > 1.2:
        insights.append("High amplitude — possible stress or muscle artifact.")
    elif power < 0.2:
        insights.append("Low amplitude — possibly calm or poor electrode contact.")

    if pred_value > 0.8:
        insights.append("⚠ Strong anomaly detected.")
    elif pred_value > 0.5:
        insights.append("Mild anomaly detected — monitor user.")
    else:
        insights.append("Signal appears normal.")

    try:
        if eeg_np[-1] - eeg_np[0] > 0.4:
            insights.append("Rising trend spike observed.")
    except Exception:
        pass

    if np.std(eeg_np) > 1.0:
        insights.append("High variability — irregular brainwave pattern.")

    return " | ".join(insights)


def grad_cam_1d(model, input_data):
    try:
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        if not model.built:
            model.build(input_data.shape)
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            return np.zeros(input_data.shape[1], dtype=float)
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(input_tensor)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_output)
        if grads is None:
            return np.zeros(input_data.shape[1], dtype=float)
        guided_grads = tf.reduce_mean(grads, axis=1)
        conv_output_np = conv_output.numpy()[0]
        guided_grads_np = guided_grads.numpy()[0]
        cam = np.dot(conv_output_np, guided_grads_np)
        cam = np.maximum(cam, 0)
        if np.max(cam) - np.min(cam) < 1e-8:
            return np.zeros_like(cam)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-12)
        cam_resized = np.interp(
            np.arange(input_data.shape[1]),
            np.linspace(0, input_data.shape[1] - 1, len(cam)),
            cam,
        )
        return cam_resized
    except Exception:
        traceback.print_exc()
        try:
            return np.zeros(input_data.shape[1], dtype=float)
        except Exception:
            return np.zeros(1, dtype=float)


def generate_report(
    eeg_values_window, pred, insight_text, heatmap=None, shap_values=None
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_name = f"report_{timestamp}.png"
    pdf_name = f"report_{timestamp}.pdf"
    png_path = os.path.join(REPORT_DIR, png_name)
    pdf_path = os.path.join(REPORT_DIR, pdf_name)

    if len(eeg_values_window) == 0:
        y = [0]
        x = [0]
    else:
        if isinstance(eeg_values_window[0], dict) and "values" in eeg_values_window[0]:
            arr = np.array([np.array(d["values"]) for d in eeg_values_window])
            y = np.mean(arr, axis=1)
            x = list(range(len(y)))
        else:
            if isinstance(eeg_values_window[0], (list, np.ndarray)):
                arr = np.array(eeg_values_window)
                if arr.ndim == 1:
                    y = arr.tolist()
                    x = list(range(len(y)))
                else:
                    y = np.mean(arr, axis=0).tolist()
                    x = list(range(len(y)))
            else:
                flat = np.ravel(eeg_values_window).astype(float).tolist()
                y = flat
                x = list(range(len(y)))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=1.2, label="EEG Waveform")

    if heatmap is not None:
        try:
            heatmap_resized = np.interp(
                np.arange(len(y)), np.linspace(0, len(y) - 1, len(heatmap)), heatmap
            )
            plt.plot(
                x,
                heatmap_resized,
                alpha=0.5,
                linestyle="--",
                label="GradCAM Importance",
            )
        except Exception:
            pass

    plt.legend()
    plt.title("EEG Report Snapshot (with Explainability)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    try:
        if shap_values is not None and feature_names is not None:
            sv = np.array(shap_values).reshape(-1)
            top_idx = np.argsort(np.abs(sv))[-5:][::-1]
            shap_text = "\n".join(
                [f"{feature_names[i]}: {abs(sv[i]):.3f}" for i in top_idx]
            )
            plt.gcf().text(0.72, 0.45, f"Top SHAP Factors:\n{shap_text}", fontsize=8)
    except Exception:
        pass

    info = f"Prediction: {pred:.3f}\nInsights: {insight_text}\nGenerated: {datetime.utcnow().isoformat()}Z"
    plt.gcf().text(0.01, 0.01, info, fontsize=9)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    return png_name, pdf_name


def send_email_report(to_email, subject, body, pdf_filename=None):
    if not SMTP_EMAIL or not SMTP_PASS:
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg.set_content(body)
    if pdf_filename:
        pdf_path = os.path.join(REPORT_DIR, pdf_filename)
        try:
            with open(pdf_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype="application",
                    subtype="pdf",
                    filename=pdf_filename,
                )
        except Exception:
            traceback.print_exc()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SMTP_EMAIL, SMTP_PASS)
            smtp.send_message(msg)
    except Exception:
        traceback.print_exc()


def send_sms_alert_text():
    if client is None:
        return
    try:
        client.messages.create(
            body="⚠ NeuroGuard AI Alert: Abnormal EEG detected. Report has been generated and emailed.",
            from_=TWILIO_NUMBER,
            to=ALERT_NUMBER,
        )
    except Exception:
        traceback.print_exc()


def send_report_mms(to_phone, png_filename):
    if client is None or not PUBLIC_BASE_URL:
        return
    media_url = f"{PUBLIC_BASE_URL.rstrip('/')}/reports/{png_filename}"
    try:
        client.messages.create(
            body="⚠ NeuroGuard AI — EEG anomaly report (image attached).",
            from_=TWILIO_NUMBER,
            to=to_phone,
            media_url=[media_url],
        )
    except Exception:
        traceback.print_exc()


def load_and_preprocess_data():
    global eeg_features_df, eeg_labels, X_train, X_test, y_train, y_test
    global normal_indices, anomaly_indices, scaler, feature_names

    df = pd.read_csv("data/emotions.csv")
    df["label"] = (
        df["label"]
        .map(
            {
                "NEUTRAL": 0,
                "POSITIVE": 1,
                "POSITIVE1": 2,
                "POSITIVE2": 3,
                "POSITIVE3": 4,
                "POSITIVE4": 5,
                "POSITIVE5": 6,
            }
        )
        .fillna(0)
    )

    eeg_labels = df["label"].values
    eeg_features_df = df.drop("label", axis=1)
    feature_names = eeg_features_df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(eeg_features_df.values)
    eeg_features_df = pd.DataFrame(X_scaled, columns=eeg_features_df.columns)

    X = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, eeg_labels, test_size=0.2, random_state=42, stratify=eeg_labels
    )
    normal_indices = np.where(eeg_labels == 0)[0]
    anomaly_indices = np.where(eeg_labels != 0)[0]


def train_cnn_model():
    global cnn_model, X_train, y_train, X_test, y_test

    cnn_model = Sequential(
        [
            Conv1D(32, 5, activation="relu", input_shape=(X_train.shape[1], 1)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            Conv1D(64, 5, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            Conv1D(64, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.4),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    y_train_binary = (y_train != 0).astype(int)

    class_weights_arr = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train_binary), y=y_train_binary
    )
    class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}

    optimizer = Adam(learning_rate=0.001)
    cnn_model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
    )

    cnn_model.fit(
        X_train,
        y_train_binary,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    return cnn_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/eeg-stream")
def eeg_stream():
    idx = np.random.choice(
        normal_indices if np.random.random() < 0.85 else anomaly_indices
    )
    row_vals = eeg_features_df.iloc[idx].values.tolist()
    try:
        dataBuffer_server.append({"time": len(dataBuffer_server), "values": row_vals})
        if len(dataBuffer_server) > DATA_BUFFER_MAX:
            dataBuffer_server.pop(0)
    except Exception:
        pass
    return jsonify({"data": row_vals, "actual_label": int(eeg_labels[idx])})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        eeg_data = data["data"]
        actual_label = data.get("actual_label")
        if cnn_model is None:
            raise RuntimeError("cnn_model is None")

        input_arr = np.array(eeg_data).reshape(1, -1, 1)
        pred = float(cnn_model.predict(input_arr, verbose=0)[0][0])
        has_disease = pred > 0.5

        if actual_label is not None:
            disease_label = int(actual_label)
            disease_name = DISEASE_LABELS.get(disease_label, "UNKNOWN")
            has_disease = disease_label != 0
        else:
            disease_label = 1 if has_disease else 0
            disease_name = DISEASE_LABELS.get(disease_label, "UNKNOWN")

        try:
            gradcam_heatmap = grad_cam_1d(cnn_model, input_arr)
            top_feature_indices = np.argsort(gradcam_heatmap)[-5:][::-1].tolist()
            top_feature_names = [feature_names[i] for i in top_feature_indices]
            top_feature_importance = [
                float(gradcam_heatmap[i]) for i in top_feature_indices
            ]
        except Exception:
            gradcam_heatmap = None
            top_feature_indices = []
            top_feature_names = []
            top_feature_importance = []

        try:
            shap_out = shap_explainer.shap_values(input_arr)
            if isinstance(shap_out, list):
                shap_vals = np.array(shap_out[0])
            else:
                shap_vals = np.array(shap_out)
            shap_vals = shap_vals.reshape(-1)
            abs_values = np.abs(shap_vals)
            top_idx = abs_values.argsort()[-5:][::-1]
            top_shap_feature_names = [feature_names[i] for i in top_idx]
            top_shap_importance = [float(abs_values[i]) for i in top_idx]
        except Exception:
            shap_vals = None
            top_shap_feature_names = []
            top_shap_importance = []
            top_idx = np.array([])

        response = {
            "has_disease": has_disease,
            "confidence": float(pred),
            "disease_label": disease_label,
            "disease_name": disease_name,
            "explainability": {
                "gradcam": (
                    gradcam_heatmap.tolist() if (gradcam_heatmap is not None) else None
                ),
                "gradcam_top_indices": top_feature_indices,
                "gradcam_top_features": top_feature_names,
                "gradcam_top_importance": top_feature_importance,
                "shap_top_indices": top_idx.tolist() if shap_vals is not None else [],
                "shap_top_features": top_shap_feature_names,
                "shap_top_importance": top_shap_importance,
                "shap_values": shap_vals.tolist() if shap_vals is not None else None,
            },
        }

        if has_disease:
            insight_text = generate_insights(pred, eeg_data)
            try:
                png, pdf = generate_report(
                    eeg_data,
                    pred,
                    insight_text,
                    heatmap=gradcam_heatmap,
                    shap_values=shap_vals,
                )
            except Exception:
                traceback.print_exc()
                png, pdf = (None, None)

            contrib_lines = ""
            if top_shap_feature_names:
                for name, score in zip(top_shap_feature_names, top_shap_importance):
                    contrib_lines += f"{name} (importance: {score:.4f})\n"
            elif top_feature_names:
                for name, score in zip(top_feature_names, top_feature_importance):
                    contrib_lines += f"{name} (gradcam importance: {score:.4f})\n"
            else:
                contrib_lines = "No explainability available."

            email_body = (
                "Dear Doctor,\n\n"
                "An abnormal EEG pattern has been detected by NeuroGuard AI.\n\n"
                f"🧠 Disease Detected: {disease_name}\n"
                f"📊 Confidence Score: {pred:.4f}\n"
                f"⏱ Timestamp: {datetime.utcnow().isoformat()}Z\n\n"
                f"🔍 Insights:\n{insight_text}\n\n"
                "📌 Top contributing EEG features (SHAP-based):\n"
                f"{contrib_lines}\n"
                "The attached PDF contains detailed explainability charts.\n\n"
                "Regards,\nNeuroGuard AI System"
            )

            try:
                send_email_report(
                    DOCTOR_EMAIL,
                    f"⚠ NeuroGuard AI — {disease_name} Pattern Detected",
                    email_body,
                    pdf_filename=pdf,
                )
            except Exception:
                traceback.print_exc()

            try:
                send_sms_alert_text()
            except Exception:
                pass

            try:
                if DOCTOR_PHONE and PUBLIC_BASE_URL and (png is not None):
                    send_report_mms(DOCTOR_PHONE, png)
            except Exception:
                pass

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc().splitlines()[-10:],
                }
            ),
            500,
        )


@app.route("/insights", methods=["POST"])
def insights():
    try:
        data = request.get_json()
        eeg_data = data["data"]
        eeg_array = np.array(eeg_data).reshape(1, -1, 1)
        pred = float(cnn_model.predict(eeg_array, verbose=0)[0][0])
        insight_text = generate_insights(pred, eeg_data)
        return jsonify({"prediction": float(pred), "insight": insight_text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/report-now", methods=["POST"])
def report_now():
    try:
        if len(dataBuffer_server) == 0:
            return jsonify({"error": "No data in server buffer"}), 400

        window = dataBuffer_server[-120:]
        last_vals = window[-1]["values"]
        eeg_array = np.array(last_vals).reshape(1, -1, 1)
        pred = float(cnn_model.predict(eeg_array, verbose=0)[0][0])
        insight_text = generate_insights(pred, last_vals)
        png_fn, pdf_fn = generate_report(window, pred, insight_text)

        try:
            send_email_report(
                DOCTOR_EMAIL,
                "NeuroGuard Manual Report",
                f"Manual report. Prediction: {pred:.3f}\n\n{insight_text}",
                pdf_fn,
            )
        except Exception:
            traceback.print_exc()

        if DOCTOR_PHONE and PUBLIC_BASE_URL and png_fn:
            send_report_mms(DOCTOR_PHONE, png_fn)

        return jsonify(
            {"status": "ok", "message": "Report generated & sent", "prediction": pred}
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/reports/<filename>")
def serve_reports(filename):
    return send_from_directory(REPORT_DIR, filename)


if _name_ == "_main_":
    print("\n=== NEUROGUARD AI STARTUP (merged) ===\n")
    load_and_preprocess_data()
    cnn_model = train_cnn_model()
    dummy_input = np.zeros((1, X_train.shape[1], 1), dtype=float)
    cnn_model.predict(dummy_input)
    try:
        _ = cnn_model.predict(np.zeros((1, X_train.shape[1], 1)))
    except Exception:
        pass

    try:
        print("\nInitializing Deep SHAP explainer (background=50)...")
        bg_size = 50
        background = X_train[
            np.random.choice(
                len(X_train), size=min(bg_size, len(X_train)), replace=False
            )
        ]
        shap_explainer = shap.DeepExplainer(cnn_model, background)
        print("Deep SHAP explainer ready.\n")
    except Exception as e:
        traceback.print_exc()
        shap_explainer = None

    print("\n🚀 Server running at: http://127.0.0.1:5000\n")
    app.run(debug=True)
