import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import noisereduce as nr
import gradio as gr
import urllib.request

class_names = ['ปกติ', 'สนิม', 'แตก']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลดโมเดล ResNet-50
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model_path = 'pump_resnet50_v1.pth'

# ลิงก์ Direct Download จาก Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1MSsQUOhuHEoPaWEIxBkJ5SGF3VJyoOz3"

if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, model_path)

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

model = model.to(device)
model.eval()

def predict_audio_file(audio_path, pump_type):
    if audio_path is None:
        return "⚠️ กรุณาอัปโหลดหรือบันทึกไฟล์เสียงก่อนกดตรวจสอบ"
    try:
        file_name = os.path.basename(audio_path)
        y, sr = librosa.load(audio_path, sr=None)

        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.61, stationary=True)

        S = librosa.feature.melspectrogram(y=y_denoised, sr=sr, n_mels=224, fmax=sr/2)
        S_dB = librosa.power_to_db(S, ref=np.max)

        cm = plt.get_cmap('magma')
        S_dB_01 = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
        rgba_img = cm(S_dB_01)

        img_array = (rgba_img[:, :, :3] * 255).astype(np.uint8)
        img_array = np.flipud(img_array)
        img = Image.fromarray(img_array).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, preds = torch.max(probabilities, dim=0)

        result = class_names[preds.item()]
        confidence_score = confidence.item() * 100

        report_text = f"📂 ไฟล์ที่ตรวจสอบ: {file_name}\n"
        report_text += f"⚙️ ประเภทปั๊มน้ำ: {pump_type}\n"
        report_text += f"📊 ความมั่นใจของ AI: {confidence_score:.2f}%\n"
        report_text += "=========================================\n\n"

        if result == "ปกติ":
            report_text += "🟢 ผลการตรวจพบ: ไม่พบความผิดปกติของตลับลูกปืน\n"
            report_text += "💡 คำแนะนำ: ทำการบำรุงรักษาตามรอบเวลามาตรฐาน (Routine Maintenance)\n"
        elif result == "สนิม":
            report_text += "🟡 ผลการตรวจพบ: ตลับลูกปืน (Bearing) เริ่มเกิดคราบสนิมหรือขาดสารหล่อลื่น\n"
            report_text += "💡 คำแนะนำ: ควรเติมสารหล่อลื่น และวางแผนการบำรุงรักษาก่อนเสียหายหนัก\n"
        elif result == "แตก":
            report_text += "🔴 ผลการตรวจพบ: ตลับลูกปืน (Bearing) เกิดการแตกหัก ชำรุดรุนแรง\n"
            report_text += "💡 คำแนะนำ: ควรหยุดการทำงานของปั๊มน้ำทันที และทำการตรวจเช็คโดยด่วน\n"

        return report_text

    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดในการประมวลผลเสียง: {e}"

web_interface = gr.Interface(
    fn=predict_audio_file,
    inputs=[
        gr.Audio(type="filepath", label="📢 อัปโหลดไฟล์เสียงปั๊มน้ำ (.mp3, .wav)"),
        gr.Dropdown(
            choices=["ปั๊มน้ำขนาดเล็ก (V.1 - ปัจจุบัน)"],
            value="ปั๊มน้ำขนาดเล็ก (V.1 - ปัจจุบัน)",
            label="🛠️ เลือกประเภท/ขนาดของปั๊มน้ำ"
        )
    ],
    outputs=gr.Textbox(label="🔍 รายงานผลการวิเคราะห์สภาวะปั๊มน้ำ", lines=10),
    title="🤖 ระบบตรวจสภาพปั๊มน้ำด้วย AI (Predictive Maintenance)",
    description="วิเคราะห์สภาวะความผิดปกติของตลับลูกปืน (Bearing Fault Detection)"
)

if __name__ == "__main__":
    web_interface.launch(server_name="0.0.0.0", server_port=7860)
