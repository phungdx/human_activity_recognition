import pandas as pd
import cv2
import mediapipe as mp

# read frames from webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# initialize mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = ['HANDSWING', 'BODYSWING']
no_of_frames = 600

def make_landmark_timesteps(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, c = frame.shape
        cx, cy = int(lm.x + w), int(lm.y + h)
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return frame


while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        # pose detection
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            # get joints estimations
            lm = make_landmark_timesteps(results)
            lm_list.append(lm)
            # draw joints
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow('img', frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label[1] + '.txt')

cap.release()
cv2.destroyAllWindows()