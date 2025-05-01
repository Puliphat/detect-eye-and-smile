import cv2
from flask import Flask, render_template, request, jsonify, redirect
import os
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import gc

app = Flask(__name__)

# โหลดโมเดล
model_eyes = tf.keras.models.load_model(r'C:\Users\Admin\Desktop\Final Project_V.1.7\Model_Eyes_98-4.h5')
model_smile = tf.keras.models.load_model(r'C:\Users\Admin\Desktop\Final Project_V.1.7\Model_Smile08-0.93.h5')

# โหลด Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ฟังก์ชันสำหรับตัดภาพใบหน้าและดวงตา
def crop_and_predict(image_path):
    try:
        # อ่านภาพต้นฉบับ
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError("ไม่สามารถอ่านไฟล์ภาพได้")

        # อ่านและปรับขนาดภาพใบหน้า
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            raise ValueError("ไม่พบใบหน้าในภาพ")
        
        # ตัดใบหน้าและดวงตา
        for (x,y,w,h) in faces:
            cv2.rectangle(original_img, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = original_img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # ทำนายภาพใบหน้า
        face_img = cv2.resize(roi_color, (64, 64))
        predictions_face = model_smile.predict(keras_image.img_to_array(face_img).reshape(1, 64, 64, 3))
        predicted_class_face = np.argmax(predictions_face)
        class_names_face = ['Smile', 'Non_Smile']

        # ทำนายภาพดวงตาที่ 1
        eye_img = cv2.resize(roi_color[y:y+h, x:x+w], (224, 224))
        predictions_eye1 = model_eyes.predict(keras_image.img_to_array(eye_img).reshape(1, 224, 224, 3))
        predicted_class_eye1 = np.argmax(predictions_eye1)
        class_names_eye = ['Open', 'Closed']

        # ทำนายภาพดวงตาที่ 2
        predictions_eye2 = model_eyes.predict(keras_image.img_to_array(eye_img).reshape(1, 224, 224, 3))
        predicted_class_eye2 = np.argmax(predictions_eye2)

        # แปลงภาพเป็น base64
        _, buffer = cv2.imencode('.jpg', original_img)
        image_base64 = base64.b64encode(buffer).decode()

        # ปล่อยทรัพยากร
        del original_img, img, gray, roi_gray, roi_color, face_img, eye_img
        gc.collect()

        # แสดงภาพต้นฉบับพร้อมผลทำนาย
        result = {
            'image': image_base64,
            'smile_prediction': class_names_face[predicted_class_face],
            'eye1_prediction': class_names_eye[predicted_class_eye1],
            'eye2_prediction': class_names_eye[predicted_class_eye2]
        }

        return result
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # ตรวจสอบว่ามีส่วนของไฟล์หรือไม่
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # หากผู้ใช้ไม่ได้เลือกไฟล์ บราวเซอร์จะส่งไฟล์ว่างๆที่ไม่มีชื่อไฟล์
        if file.filename == '':
            return redirect(request.url)
        if file:
            try:
                # สร้าง uploads directory ถ้ายังไม่มี
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                
                # บันทึกไฟล์ที่อัปโหลดไว้ที่ตำแหน่งชั่วคราว
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)

                # เรียกใช้ฟังก์ชัน crop_and_predict
                prediction_result = crop_and_predict(file_path)

                # ปิดการใช้งานไฟล์ทั้งหมด
                gc.collect()

                # ลบไฟล์ที่อัปโหลด
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {file_path}: {str(e)}")
                    # ไม่ต้อง return error เพราะการลบไฟล์ไม่ใช่ฟังก์ชันหลัก

                return jsonify(prediction_result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
