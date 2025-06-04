import cv2
import numpy as np

MASK_FILES = [
    "mask1.png",
    "mask2.png",
    "mask3.png",
    "mask4.png",
    "mask5.png"
]
MASK_IMAGES = [cv2.imread(m, cv2.IMREAD_UNCHANGED) for m in MASK_FILES]


def overlay_image():
    global image
    # Wczytaj obrazek do nałożenia (może być z przezroczystością - 4 kanały)
    overlay = cv2.imread(r'C:\Users\trape\GitHub\WMA_PL_dzienne_macmac\cw5\pliki\overlay.png', cv2.IMREAD_UNCHANGED)

    # Pozycja overlaya z trackbara (np. x = ksize, y = low)
    x = cv2.getTrackbarPos('ksize', 'obrazek')
    y = cv2.getTrackbarPos('low', 'obrazek')

    # Rozmiar overlaya z trackbara (np. high to szerokość, ksize to wysokość)
    w = cv2.getTrackbarPos('high', 'obrazek')
    h = cv2.getTrackbarPos('ksize', 'obrazek')
    if w == 0: w = overlay.shape[1]
    if h == 0: h = overlay.shape[0]

    img_overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # Nakładanie
    img_result = image.copy()
    img_result = overlay_image_alpha(img_result, img_overlay, (x, y))
    cv2.imshow('obrazek', img_result)

def overlay_image_alpha(img, img_overlay, pos):
    """Nałóż obrazek z kanałem alfa na drugi obrazek (z przezroczystością)"""
    x, y = pos
    # Jeśli overlay ma kanał alfa
    if img_overlay.shape[2] == 4:
        b, g, r, a = cv2.split(img_overlay)
        overlay_color = cv2.merge((b, g, r))
        mask = a
    else:
        overlay_color = img_overlay
        # Jeśli brak kanału alfa, to przezroczystość zero tam gdzie białe
        gray = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    h, w = overlay_color.shape[:2]
    # Ograniczenie rozmiaru, aby nie wyjść poza obrazek
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
        overlay_color = overlay_color[:, :w]
        mask = mask[:, :w]
    if y + h > img.shape[0]:
        h = img.shape[0] - y
        overlay_color = overlay_color[:h, :]
        mask = mask[:h, :]

    roi = img[y:y + h, x:x + w]
    mask_3ch = cv2.merge([mask, mask, mask])
    img_bg = cv2.bitwise_and(roi, 255 - mask_3ch)
    img_fg = cv2.bitwise_and(overlay_color, mask_3ch)
    dst = cv2.add(img_bg, img_fg)
    img[y:y + h, x:x + w] = dst
    return img

class FaceTracker:
    def __init__(self, max_lost=10):
        self.next_id = 0
        self.faces = dict()
        self.max_lost = max_lost

    def update(self, detections):
        new_faces = dict()
        used = set()
        for fid, data in self.faces.items():
            min_dist = float('inf')
            min_idx = -1
            for i, (x, y, w, h) in enumerate(detections):
                if i in used:
                    continue
                cx, cy = x + w // 2, y + h // 2
                dist = np.hypot(data['centroid'][0] - cx, data['centroid'][1] - cy)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_idx != -1 and min_dist < 80:
                x, y, w, h = detections[min_idx]
                new_faces[fid] = {
                    'bbox': (x, y, w, h),
                    'centroid': (x + w // 2, y + h // 2),
                    'lost': 0,
                    'mask_id': data['mask_id']
                }
                used.add(min_idx)
            else:
                data['lost'] += 1
                if data['lost'] < self.max_lost:
                    new_faces[fid] = data
        for i, (x, y, w, h) in enumerate(detections):
            if i not in used:
                mask_id = self.next_id % len(MASK_IMAGES)
                new_faces[self.next_id] = {
                    'bbox': (x, y, w, h),
                    'centroid': (x + w // 2, y + h // 2),
                    'lost': 0,
                    'mask_id': mask_id
                }
                self.next_id += 1
        self.faces = new_faces

    def get_faces(self):
        return self.faces

def nothing(x):
    pass

print("Wybierz źródło obrazu:")
print("1 - Kamera")
print("2 - Plik video (np. film.mp4)")
choice = 2

if choice == "1":
    cap = cv2.VideoCapture(0)
else:
    filename = "5 People 1 Guitar!.mp4"
    cap = cv2.VideoCapture(filename)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tracker = FaceTracker()

# --- DODANE TRACKBARY ---
cv2.namedWindow("Face masks live")
cv2.createTrackbar('scale', 'Face masks live', 11, 20, nothing)   # scaleFactor x0.1 (1.1..2.0)
cv2.createTrackbar('minN', 'Face masks live', 8, 20, nothing)     # minNeighbors (1..20)
cv2.createTrackbar('minS', 'Face masks live', 100, 300, nothing)  # minSize (px)
cv2.createTrackbar('offsetY', 'Face masks live', 50, 100, nothing) # -50..+50 (środek suwaka = 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- ODCZYT TRACKBARÓW ---
    scale = cv2.getTrackbarPos('scale', 'Face masks live') / 10.0
    minN = cv2.getTrackbarPos('minN', 'Face masks live')
    minS = cv2.getTrackbarPos('minS', 'Face masks live')
    offsetY = cv2.getTrackbarPos('offsetY', 'Face masks live') - 50  # środek = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=minN,
        minSize=(minS, minS)
    )
    tracker.update(faces)
    tracked = tracker.get_faces()

    for fid, data in tracked.items():
        x, y, w, h = data['bbox']
        mask_id = data['mask_id']
        mask_img = MASK_IMAGES[mask_id]
        mask_width, mask_height, mask_x, mask_y = 0, 0, 0, 0
        # Dobierz miejsce nakładania dla każdej maski osobno (z offsetY):
        match mask_id:  # mask1: jeszcze wyżej!
            case 1:
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = max(0, y - mask_height + 10 + offsetY)
            case 2:  # mask2, mask4: na głowie standardowo
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = max(0, y - mask_height + 10 + offsetY)
            case 3: # mask3: na oczach
                mask_width = int(w * 1.03)
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x - int(w * 0.015)
                mask_y = y + int(h * 0.08) + offsetY
            case 4:  # mask5: wąsy
                mask_width = int(w * 0.75)
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x + int(w * 0.125)
                mask_y = y + int(h * 0.52) + offsetY
            case _:  # domyślna pozycja (na czole)
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = y + offsetY
        frame = overlay_image_alpha(frame, mask_img, (mask_x, mask_y))
        # Ramka i podpis
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
        cv2.putText(frame, f'ID:{fid} mask:{mask_id+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Face masks live", frame)
    key = cv2.waitKey(1)
    if key == 27: break

cap.release()
cv2.destroyAllWindows()