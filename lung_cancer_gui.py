# 'import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
# import tensorflow as tf
# import cv2
# import os
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.units import inch

# # Load trained model
# model = tf.keras.models.load_model("lung_multiclass_model.h5")

# classes = list(os.listdir("dataset"))

# # GradCAM
# def generate_gradcam(img_array):
#     last_conv_layer = model.get_layer("Conv_1")

#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [last_conv_layer.output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, np.argmax(predictions[0])]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# # PDF Generator
# def generate_pdf(result_text):
#     doc = SimpleDocTemplate("Lung_Cancer_AI_Report.pdf")
#     styles = getSampleStyleSheet()
#     elements = []

#     elements.append(Paragraph("<b>Lung Cancer AI Diagnostic Report</b>", styles["Title"]))
#     elements.append(Spacer(1, 20))
#     elements.append(Paragraph(result_text.replace("\n","<br/>"), styles["Normal"]))
#     elements.append(Spacer(1, 20))

#     if os.path.exists("gradcam_output.jpg"):
#         elements.append(RLImage("gradcam_output.jpg", width=4*inch, height=4*inch))

#     doc.build(elements)

# # Prediction
# def predict_image():
#     file_path = filedialog.askopenfilename()

#     img = Image.open(file_path).resize((224,224))
#     img_array = np.array(img)/255.0
#     img_exp = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_exp)
#     class_index = np.argmax(prediction)
#     confidence = float(np.max(prediction)*100)

#     predicted_class = classes[class_index]

#     if predicted_class.lower() == "normal":
#         status = "No Cancer Detected"
#         severity = "None"
#         advice = "Lungs appear normal. Maintain healthy lifestyle."
#     else:
#         status = "Cancer Detected"
#         if confidence < 60:
#             severity = "Early Stage"
#         elif confidence < 85:
#             severity = "Moderate Stage"
#         else:
#             severity = "Advanced Stage"
#         advice = "Consult oncologist immediately for further examination."

#     # GradCAM
#     heatmap = generate_gradcam(img_exp)
#     heatmap = cv2.resize(heatmap, (224,224))
#     heatmap = np.uint8(255*heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     superimposed = cv2.addWeighted((img_array*255).astype("uint8"), 0.6, heatmap, 0.4, 0)
#     cv2.imwrite("gradcam_output.jpg", superimposed)

#     result_text = f"""
# Prediction: {predicted_class}
# Confidence Score: {confidence:.2f} %

# Cancer Status: {status}
# Severity Level: {severity}

# Medical Advice:
# {advice}
# """

#     result_label.config(text=result_text)
#     generate_pdf(result_text)

# # GUI
# root = tk.Tk()
# root.title("AI-Based Lung Cancer Detection System")
# root.geometry("650x550")
# root.configure(bg="#f4f6f7")

# title = tk.Label(root, text="AI Lung Cancer Detection System",
#                  font=("Helvetica",18,"bold"), bg="#f4f6f7")
# title.pack(pady=15)

# btn = tk.Button(root, text="Upload CT Scan",
#                 font=("Helvetica",12,"bold"),
#                 bg="#27ae60", fg="white",
#                 width=20, height=2,
#                 command=predict_image)
# btn.pack(pady=10)

# result_label = tk.Label(root, text="", justify="left",
#                         font=("Helvetica",11),
#                         bg="#f4f6f7")
# result_label.pack(pady=20)

# root.mainloop()
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Load trained model
model = tf.keras.models.load_model("lung_multiclass_model.h5")

# Class labels
classes = list(os.listdir("dataset"))

# ---------------------- GradCAM FUNCTION ----------------------
def generate_gradcam(img_array):
    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ---------------------- PDF GENERATOR ----------------------
def generate_pdf(result_text):
    doc = SimpleDocTemplate("Lung_Cancer_AI_Report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Lung Cancer AI Diagnostic Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(result_text.replace("\n","<br/>"), styles["Normal"]))
    elements.append(Spacer(1, 20))

    if os.path.exists("gradcam_output.jpg"):
        elements.append(RLImage("gradcam_output.jpg", width=4*inch, height=4*inch))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "<b>Note:</b> This is an AI-generated report. Doctor consultation is required for final diagnosis.",
        styles["Normal"]
    ))

    doc.build(elements)

# ---------------------- PREDICTION FUNCTION ----------------------
def predict_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # Load image
    img = Image.open(file_path).resize((224,224))
    img_array = np.array(img) / 255.0
    img_exp = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_exp)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    predicted_class = classes[class_index]

    # Result Logic
    if predicted_class.lower() == "normal":
        status = "No Cancer Detected"
        severity = "None"
        advice = "Lungs appear normal. Maintain healthy lifestyle."
    else:
        status = "Cancer Detected"

        if confidence < 60:
            severity = "Early Stage"
        elif confidence < 85:
            severity = "Moderate Stage"
        else:
            severity = "Advanced Stage"

        advice = "Consult oncologist immediately for further examination."

    # ---------------------- IMPROVED GRADCAM ----------------------
    heatmap = generate_gradcam(img_exp)
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)

    # Find strongest disease region
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(heatmap)

    # Create mask for single region
    mask = np.zeros_like(heatmap)
    cv2.circle(mask, maxLoc, 30, 255, -1)

    # Apply mask
    focused_heatmap = cv2.bitwise_and(heatmap, mask)
    focused_heatmap = cv2.applyColorMap(focused_heatmap, cv2.COLORMAP_JET)

    # Overlay
    original_img = (img_array * 255).astype("uint8")
    superimposed = cv2.addWeighted(original_img, 0.7, focused_heatmap, 0.6, 0)

    # Draw circle (highlight region)
    cv2.circle(superimposed, maxLoc, 35, (0,255,0), 2)

    cv2.imwrite("gradcam_output.jpg", superimposed)

    # ---------------------- RESULT TEXT ----------------------
    result_text = f"""
Prediction: {predicted_class}
Confidence Score: {confidence:.2f} %

Cancer Status: {status}
Severity Level: {severity}

Medical Advice:
{advice}
"""

    result_label.config(text=result_text)

    # Generate PDF
    generate_pdf(result_text)

# ---------------------- GUI ----------------------
root = tk.Tk()
root.title("AI-Based Lung Cancer Detection System")
root.geometry("650x550")
root.configure(bg="#f4f6f7")

title = tk.Label(root, text="AI Lung Cancer Detection System",
                 font=("Helvetica",18,"bold"), bg="#f4f6f7")
title.pack(pady=15)

btn = tk.Button(root, text="Upload CT Scan",
                font=("Helvetica",12,"bold"),
                bg="#27ae60", fg="white",
                width=20, height=2,
                command=predict_image)
btn.pack(pady=10)

result_label = tk.Label(root, text="", justify="left",
                        font=("Helvetica",11),
                        bg="#f4f6f7")
result_label.pack(pady=20)

root.mainloop()