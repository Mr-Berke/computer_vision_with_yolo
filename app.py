import os
import cv2
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import threading
import numpy as np
import cvzone
import math
from sort import *

model = YOLO('../yolo_basic/yolov8l.pt')
model2 = YOLO('../yolo_basic/yolov8n.pt')

class_name = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "sofa", "potted plant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class Processor:
    def __init__(self):
        self.file_path = None

    def set_file_path(self, file_path):
        self.file_path = file_path
    
    def find_plate_number(self):
        file_path = self.file_path

    def detection_object(self):
        if self.file_path:
            results = model(self.file_path, show=True)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  

    def car_counter(self):
        # video için
        capture = cv2.VideoCapture("C:/Users/Berke/Desktop/jupyter/.pyler/yolo3/videos/araba.mp4")

        # video için maske
        mask = cv2.imread("C:/Users/Berke/Desktop/jupyter/.pyler/yolo3/images/mask.jpg")

        # sayma
        traker = Sort(max_age=50, min_hits=2, iou_threshold=0.3)

        # çizgi kordinatları
        limits = [380, 425, 1180, 425]
        counter = []

        while True:
            success, img = capture.read()
            img_region = cv2.bitwise_and(img, mask)
            if not success:
                print("görüntü okunmadı.")
                break

            results = model(img_region, stream=True)

            detections = np.empty((0, 5))

            for i in results:
                boxes = i.boxes
                for box in boxes:
                    # yüzdelik kutusu
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # yükseklik ve genişlik
                    w, h = x2 - x1, y2 - y1

                    # yüzde hesabı
                    conf = math.ceil(box.conf[0] * 100)

                    # nesne isimleri
                    cls = int(box.cls[0])
                    currentclass = class_name[cls]

                    if currentclass in ["car", "motorcycle", "truck", "bus"] and conf >= 30:
                        current_array = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, current_array))

            results_trackers = traker.update(detections)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

            for results in results_trackers:
                x1, y1, x2, y2, (Id) = results
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0), colorC=(0, 255, 255))

                center_x, center_y = int(x1 + w / 2), int(y1 + h / 2)
                cv2.circle(img, (center_x, center_y), 5, color=(255, 0, 0))

                if limits[0] < center_x < limits[2] and limits[1] - 40 < center_y < limits[1] + 40:
                    if counter.count(Id) == 0:
                        counter.append(Id)

            cvzone.putTextRect(img, f'Toplam: {len(counter)} ', (50, 50), colorR=(255, 0, 0))

            cv2.imshow('Video', img)
            #cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        capture.release()
        cv2.destroyAllWindows()

    def car_counter2(self):
        capture = cv2.VideoCapture("C:/Users/Berke/Desktop/jupyter/.pyler/yolo3/videos/trafik3.mp4")

        mask = cv2.imread("C:/Users/Berke/Desktop/jupyter/.pyler/yolo3/images/mask3.jpg")

        traker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        limits = [150,545, 630, 545]
        counter = []

        while True:
            success, img = capture.read()
            if not success:
                print("görüntü okunmadı.")
                break

            img_region = cv2.bitwise_and(img, mask)
            results = model(img_region, stream=True)

            detections = np.empty((0, 5))

            for i in results:
                boxes = i.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil(box.conf[0] * 100)
                    cls = int(box.cls[0])
                    currentclass = class_name[cls]

                    if currentclass in ["car", "motorcycle", "truck", "bus"] and conf >= 30:
                        current_array = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, current_array))

            results_trackers = traker.update(detections)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

            for result in results_trackers:
                x1, y1, x2, y2, Id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0), colorC=(0, 255, 255))

                center_x, center_y = int(x1 + w / 2), int(y1 + h / 2)
                cv2.circle(img, (center_x, center_y), 5, color=(255, 0, 0))

                # Gelişmiş çizgi kontrolü ve debugging
                if limits[0] < center_x < limits[2] and (limits[1] - 20) < center_y < (limits[1] + 20):
                    if Id not in counter:
                        counter.append(Id)
                        #print(f"Araç ID'si {Id} sayıldı.")

            cvzone.putTextRect(img, f'Toplam: {len(counter)} ', (50, 50), colorR=(255, 0, 0))

            cv2.imshow('Video', img)
            # cv2.imshow('Maskelenmis_Goruntu', img_region)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
    

    def find_plate_number(self):        

        file_path=self.file_path

        if file_path is None:
            print("Dosya yolu ayarlanmamış")
            return
        
        if not os.path.exists(file_path):
            print(f"Dosya yolu yok: {file_path}")
            return 
              
        # Haar cascade dosyasının yolu
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

        min_area = 500      # Minimum alan

        img = cv2.imread(file_path)        # Görüntüyü yükle

        if img is None:
            print("Görüntü dosyası yüklenemedi. Lütfen dosya yolunu kontrol edin.")
        
        else:
            # Gri tonlamaya çevir
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Plakaları tespit et
            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            # Tespit edilen plakaları işaretle
            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, "plaka", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

            # Görüntüyü göster
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def find_plate_number_and_save(self):

        file_path=self.file_path

        plaka_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        min_area = 400
        max_area = 5000

        save_dir = "C:/Users/Berke/Desktop/jupyter/.pyler/goruntu_islme_vscode/plakalar"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        capture = cv2.VideoCapture(file_path)

        tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
        plaka_counter = 0
        saved_ids = set()

        while True:
            ret, frame = capture.read()

            if not ret:
                print("Video okunmadı")
                break

            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plakalar = plaka_cascade.detectMultiScale(frame, 1.1, 4)

            detections = np.empty((0, 5))

            for (x, y, w, h) in plakalar:
                area = w * h
                if min_area < area < max_area:
                    detections = np.vstack((detections, [x, y, x + w, y + h, 1]))
            trackered_objects = tracker.update(detections)

            for x1, y1, x2, y2, id in trackered_objects:
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                if id not in saved_ids:
                    imgRoi = frame[y1:y2, x1:x2]
                    roi_file_name = os.path.join(save_dir, f"plaka_{plaka_counter}.png")
                    success = cv2.imwrite(roi_file_name, imgRoi)
                    plaka_counter += 1
                    if success:
                        saved_ids.add(id)

                cv2.putText(frame, f"Plaka {plaka_counter}", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            cv2.imshow("video", frame)

            # Pencereyi kapatmak için 'q' tuşuna basılmasını bekler
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def object_detection_cam(self):
        
        capture = cv2.VideoCapture(0)
        capture.set(3, 640)
        capture.set(4, 480)

        if not capture.isOpened():
            print("Kamera açılamadı.")
            return

        while True:
            success, img = capture.read()
            if not success:
                print("görüntü okunmadı.")
                break

            results = model(img, stream=True)
            for i in results:
                boxes = i.boxes
                for box in boxes:
                    # yüzdelik kutusu
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))

                    # yüzde hesabı
                    conf = math.ceil(box.conf[0] * 100)

                    # nesne isimleri
                    cls = int(box.cls[0])
                    cvzone.putTextRect(img, f'{class_name[cls]} {conf}' + "%", (max(0, x1), max(35, y1)), scale=1, thickness=2)

            cv2.imshow('Video', img)            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

            capture.release()
            cv2.destroyAllWindows()             
        
    def plate_number_cam(self):

        frame_widht = 640
        frame_hight = 480
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        min_area = 500

        count = 0

        # Kayıt klasörünü kontrol et ve oluştur
        save_dir = 'C:/Users/Berke/Desktop/jupyter/.pyler/goruntu_islme_vscode/plakalar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cap = cv2.VideoCapture(0)
        cap.set(3, frame_widht)
        cap.set(4, frame_hight)
        cap.set(10, 150)

        while True:
            success, img = cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, "plaka", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))
                    imgROI = img[y:y+h, x:x+w]  # Plaka bölgesini kesme

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                if 'imgROI' in locals():  # imgROI tanımlıysa kaydet
                    save_path = os.path.join(save_dir, "plaka_" + str(count) + ".jpg")
                    if cv2.imwrite(save_path, imgROI):
                        print(f"Plaka kaydedildi: {save_path}")
                        cv2.putText(img, "plaka kaydedildi", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                        cv2.imshow("sonuc", img)
                        cv2.waitKey(500)
                        count += 1
                    else:
                        print("Plaka kaydedilemedi!")
                else:
                    print("Plaka bulunamadı ve kaydedilemedi.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
object_processor = Processor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        object_processor.set_file_path(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})

@app.route('/process', methods=['POST'])
def process_file():
    action = request.json.get('action')
    if action == 'nesne_bul':
        threading.Thread(target=object_processor.detection_object).start()
        return jsonify({'message': 'İşlem başlatıldı'})
    elif action == 'nesne_bul_kamera':
        threading.Thread(target=object_processor.object_detection_cam).start()
        return jsonify({'message': 'İşlem başlatıldı'})
    elif action == 'arac_say':
        threading.Thread(target=object_processor.car_counter).start()
        return jsonify({'message': 'İşlem başlatıldı'})
    elif action == 'arac_say2':
        threading.Thread(target=object_processor.car_counter2).start()
        return jsonify({'message': 'İşlem başlatıldı'})
    elif action == 'plaka_bul':
        threading.Thread(target=object_processor.find_plate_number).start()
        return jsonify({'message': 'İşlem başlatıldı'})
    elif action == 'plaka_bul_kaydet':
        threading.Thread(target=object_processor.find_plate_number_and_save).start()
        return jsonify({'message': 'İşlem başlatıldı'})    
    elif action == 'plaka_bul_kaydet_kamera':
        threading.Thread(target=object_processor.plate_number_cam).start()
        return jsonify({'message': 'İşlem başlatıldı'})    
    else:
        return jsonify({'message': 'Geçersiz işlem'})
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
