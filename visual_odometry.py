import numpy as np
import cv2 

cap = cv2.VideoCapture("C:\\Karina\\s2\\pve\\eas\\FINAL PROJECT_Karina Adi Putri\\video_test24.mp4")

def extract_features(frame):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return matches

# # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (1000, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
orb = cv2.ORB_create()

height, weight, _ = old_frame.shape

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# print(p0)

cumulative_dx = 0
cumulative_dy = 0
cumulative_rotation = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    kp1, des1 = extract_features(frame_gray)


    p0 = np.float32([[m.pt] for m in kp1])

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print(p1.shape)
    # Select good points
    if p1 is not None:
        good_new = p1
        good_old = p0
        # print(f"good new: {good_new[:10]}")
        # print(f"good old: {good_old[:10]}")

        detect_x = []
        detect_y = []
        detect_rotation = []

        for i in range(p1.shape[0] - 1):
            pin_new = good_new[i, 0]
            pin_old = good_old[i, 0]
            pin2_new = good_new[i+1, 0]
            pin2_old = good_old[i+1, 0]
            # print(f"good new: {pin_new}")
            # print(f"good old: {pin_old}")

            dx = pin_new[0] - pin_old[0]
            if dx <200 and dx > -200:
                detect_x.append(dx)
            

            dy = pin_new[1] - pin_old[1]
            if dy < 100 and dy > -100:
                detect_y.append(dy)
        

            theta1 = np.arctan2((pin_old[1]-pin2_old[1]), (pin_old[0]-pin2_old[0]))
            theta2 = np.arctan2((pin_new[1]-pin2_new[1]), (pin_new[0]-pin2_old[0]))
            dr = (theta2 - theta1) * 180 / np.pi
            if dr < 15 and dr > -15:
                detect_rotation.append(dr)

        detect_x = np.mean(detect_x)
        detect_y = np.mean(detect_y)
        detect_rotation = np.mean(detect_rotation)
        cumulative_rotation += detect_rotation
        cumulative_dx += detect_x * np.cos(cumulative_rotation * np.pi/180)
        cumulative_dy += detect_y * np.cos(cumulative_rotation * np.pi/180)

        print(f"translasi: ({cumulative_dx/10:0.2f},{cumulative_dy/10:0.2f})\trotation: {cumulative_rotation:0.2f}")


    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # print(f"new2: {new}")
        # print(f"old2: {old}")
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
cv2.destroyAllWindows()