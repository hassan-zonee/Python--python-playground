import cv2
import dlib
import numpy as np
from collections import deque, Counter


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

input_video_path = 'test.mp4'
output_video_path = 'out_test.mp4'

cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_width = frame_height * 9 // 16
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, frame_height))

last_speaker_face = None
max_lip_distance = 0

speaker_history = deque(maxlen=10)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    speaking_face = None
    current_max_lip_distance = 0

    for face in faces:
        landmarks = predictor(gray, face)
        top_lip = landmarks.parts()[48]
        bottom_lip = landmarks.parts()[54]

        lip_distance = euclidean_distance((top_lip.x, top_lip.y), (bottom_lip.x, bottom_lip.y))

        if lip_distance > current_max_lip_distance:
            current_max_lip_distance = lip_distance
            speaking_face = face

    if speaking_face:
        speaker_history.append(speaking_face)

    frame_counter += 1

    if frame_counter % 10 == 0 and speaker_history:
        face_counts = Counter([(f.left(), f.top(), f.right(), f.bottom()) for f in speaker_history])
        most_common_face, count = face_counts.most_common(1)[0]

        if count >= 10:  # At least 60% consistency
            for f in speaker_history:
                if (f.left(), f.top(), f.right(), f.bottom()) == most_common_face:
                    last_speaker_face = f
                    break

    if last_speaker_face:
        (x, y, w, h) = (last_speaker_face.left(), last_speaker_face.top(),
                        last_speaker_face.width(), last_speaker_face.height())
    else:
        out.write(frame)
        continue

    face_center = (x + w // 2, y + h // 2)
    crop_x1 = max(0, face_center[0] - output_width // 2)
    crop_x2 = min(frame_width, face_center[0] + output_width // 2)

    crop_y1 = max(0, face_center[1] - frame_height // 2)
    crop_y2 = min(frame_height, face_center[1] + frame_height // 2)

    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_frame_resized = cv2.resize(cropped_frame, (output_width, frame_height))

    out.write(cropped_frame_resized)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed!")
