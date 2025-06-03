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

def overlay_image_alpha(img, img_overlay, pos, overlay_size=None):
    x, y = pos
    if overlay_size is not None:
        img_overlay = cv2.resize(img_overlay, overlay_size, interpolation=cv2.INTER_AREA)
    if img_overlay.shape[2] == 4:
        # PNG z alfą
        b, g, r, a = cv2.split(img_overlay)
        overlay_color = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a))
    else:
        # JPEG bez alfy – traktuj białe jako przezroczyste
        overlay_color = img_overlay.copy()
        gray = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2GRAY)
        _, a = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.merge((a, a, a))
    h, w = overlay_color.shape[:2]
    if y < 0:
        overlay_color = overlay_color[-y:, :, :]
        mask = mask[-y:, :, :]
        h = overlay_color.shape[0]
        y = 0
    if x < 0:
        overlay_color = overlay_color[:, -x:, :]
        mask = mask[:, -x:, :]
        w = overlay_color.shape[1]
        x = 0
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return img
    roi = img[y:y+h, x:x+w]
    img_bg = cv2.bitwise_and(roi, 255 - mask)
    img_fg = cv2.bitwise_and(overlay_color, mask)
    dst = cv2.add(img_bg, img_fg)
    img[y:y+h, x:x+w] = dst
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

print("Wybierz źródło obrazu:")
print("1 - Kamera")
print("2 - Plik video (np. film.mp4)")
choice = input("Twój wybór [1/2]: ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
else:
    filename = input("Podaj nazwę pliku wideo (np. film.mp4): ").strip()
    cap = cv2.VideoCapture(filename)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tracker = FaceTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(100, 100))
    tracker.update(faces)
    tracked = tracker.get_faces()

    for fid, data in tracked.items():
        x, y, w, h = data['bbox']
        mask_id = data['mask_id']
        mask_img = MASK_IMAGES[mask_id]
        # Dobierz miejsce nakładania dla każdej maski osobno:
        if mask_id == 0:  # mask1: jeszcze wyżej!
            mask_width = w
            mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
            mask_x = x
            mask_y = max(0, y - mask_height + 10)  # -15% wysokości twarzy nad głową
        elif mask_id in [1, 3]:  # mask2, mask4: na głowie standardowo
            mask_width = w
            mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
            mask_x = x
            mask_y = max(0, y - mask_height + 10)
        elif mask_id == 2:  # mask3: na oczach
            mask_width = int(w * 1.03)
            mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
            mask_x = x - int(w * 0.015)
            mask_y = y + int(h * 0.08)
        elif mask_id == 4:  # mask5: wąsy
            mask_width = int(w * 0.75)
            mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
            mask_x = x + int(w * 0.125)
            mask_y = y + int(h * 0.52)
        else:  # domyślna pozycja (na czole)
            mask_width = w
            mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
            mask_x = x
            mask_y = y
        frame = overlay_image_alpha(frame, mask_img, (mask_x, mask_y), (mask_width, mask_height))
        # Ramka i podpis
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
        cv2.putText(frame, f'ID:{fid} mask:{mask_id+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Face masks live", frame)
    key = cv2.waitKey(1)
    if key == 27: break

cap.release()
cv2.destroyAllWindows()