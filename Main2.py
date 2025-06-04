import cv2
import numpy as np
import os

MASK_FILES = [
    "mask1.png",
    "mask2.png",
    "mask3.png",
    "mask4.png",
    "mask5.png"
]
MASK_IMAGES = [cv2.imread(m, cv2.IMREAD_UNCHANGED) for m in MASK_FILES]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


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


def overlay_image_alpha(img, img_overlay, pos, overlay_size=None):
    x, y = pos
    if img_overlay is None:
        print("Overlay nie wczytany!")
        return img
    if overlay_size is not None:
        img_overlay = cv2.resize(img_overlay, overlay_size, interpolation=cv2.INTER_AREA)
    if img_overlay.shape[2] == 4:
        b, g, r, a = cv2.split(img_overlay)
        overlay_color = cv2.merge((b, g, r))
        mask = a
    elif img_overlay.shape[2] == 3:
        b, g, r = cv2.split(img_overlay)
        overlay_color = cv2.merge((b, g, r))
        mask = np.ones_like(b) * 255  # sztuczny alfa
    else:
        print("Nieobsługiwany format kanałów:", img_overlay.shape)
        return img
    h, w = overlay_color.shape[:2]
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
    if h <= 0 or w <= 0 or overlay_color.size == 0 or mask.size == 0:
        print("Nakładka poza obrazem lub pusta!")
        return img
    roi = img[y:y+h, x:x+w]
    mask_3ch = cv2.merge([mask, mask, mask])
    img_bg = cv2.bitwise_and(roi, 255 - mask_3ch)
    img_fg = cv2.bitwise_and(overlay_color, mask_3ch)
    dst = cv2.add(img_bg, img_fg)
    img[y:y+h, x:x+w] = dst
    return img

def detect_and_draw(frame, tracker, overlay_file=None):
    scale = cv2.getTrackbarPos('scale', 'Face masks live') / 10.0
    minN = cv2.getTrackbarPos('minN', 'Face masks live')
    minS = cv2.getTrackbarPos('minS', 'Face masks live')
    offsetY = cv2.getTrackbarPos('offsetY', 'Face masks live') - 50  # środek = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, minN, minSize=(minS, offsetY))
    tracker.update(faces)
    tracked = tracker.get_faces()

    for fid, data in tracked.items():
        x, y, w, h = data['bbox']
        mask_id = data['mask_id']
        mask_img = MASK_IMAGES[mask_id]

        # Dla czytelności: opisy masek
        mask_labels = [
            "zielona herbata",  # mask1.png
            "tecza",  # mask2.png
            "maska",  # mask3.png
            "czapka klauna",  # mask4.png
            "wasy"  # mask5.png
        ]
        mask_label = mask_labels[mask_id]

        # Każdy obrazek ma swój case:
        match mask_id:
            case 0:  # mask1.png – zielona herbata (na czole/lekko nad)
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = max(0, y - mask_height + 10 + offsetY)
            case 1:  # mask2.png – tęcza (nad głową)
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = max(0, y - mask_height + 10 + offsetY)
            case 2:  # mask3.png – maska (niżej na twarzy, wycentrowana)
                mask_width = int(w * 1.4)
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x + w // 2 - mask_width // 2  # wyśrodkowanie maski
                mask_y = y + int(h * 0.02) - 25  # dalej możesz tym regulować pionowo
            case 3:  # mask4.png – czapka klauna (na głowie)
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = max(0, y - mask_height + 10 + offsetY)
            case 4:  # mask5.png – wąsy (pod nosem)
                mask_width = int(w * 0.75)
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x + int(w * 0.125)
                mask_y = y + int(h * 0.52) + offsetY
            case _:  # awaryjnie na czole
                mask_width = w
                mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
                mask_x = x
                mask_y = y + offsetY

        frame = overlay_image_alpha(frame, mask_img, (mask_x, mask_y), (mask_width, mask_height))
        if overlay_file:  # dodatkowy overlay, np. wybrany przez użytkownika
            overlay = cv2.imread(overlay_file, cv2.IMREAD_UNCHANGED)
            if overlay is not None:
                ov_w = int(w * 1.0)
                ov_h = int(overlay.shape[0] * (ov_w / overlay.shape[1]))
                ov_x = x
                ov_y = y
                frame = overlay_image_alpha(frame, overlay, (ov_x, ov_y), (ov_w, ov_h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # PODPIS: ID i opis maski!
        cv2.putText(
            frame,
            f'ID:{fid} - {mask_label}',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
    return frame


def run_video(filename="DOOM (acapella).mp4", overlay_file=None):
    tracker = FaceTracker()
    cap = cv2.VideoCapture(0 if filename is None else filename)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = detect_and_draw(frame, tracker, overlay_file)
        cv2.imshow("Face masks live", frame)
        key = cv2.waitKey(1)
        if key == 27: break
    cap.release()
    cv2.destroyAllWindows()


def run_image(img_path, overlay_file=None):
    tracker = FaceTracker()
    image = cv2.imread(img_path)
    frame = detect_and_draw(image, tracker, overlay_file)
    cv2.imshow("Face masks live", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nothing():
    pass
if __name__ == '__main__':
    cv2.namedWindow("Face masks live")
    cv2.createTrackbar('scale', 'Face masks live', 11, 20, nothing)  # scaleFactor x0.1 (1.1..2.0)
    cv2.createTrackbar('minN', 'Face masks live', 8, 20, nothing)  # minNeighbors (1..20)
    cv2.createTrackbar('minS', 'Face masks live', 100, 300, nothing)  # minSize (px)
    cv2.createTrackbar('offsetY', 'Face masks live', 50, 100, nothing)  # -50..+50 (środek suwaka = 0)
    print("Wybierz tryb:")
    print("1 - Kamera")
    print("2 - Plik video")
    print("3 - Pojedynczy obrazek")
    print("4 - Playground (tu możesz rozwinąć dalej)")
    choice = "1" #input("Tryb [1/2/3]: ")
    if choice == "1":
        run_video()
    elif choice == "2":
        run_video()
    elif choice == "3":
        img_path = input("Podaj nazwę pliku obrazka: ")
        run_image(img_path)
    # Możesz dodać dalej swój tryb playground, albo custom tryby
