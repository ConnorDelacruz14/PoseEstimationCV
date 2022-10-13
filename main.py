import cv2 as cv
import mediapipe as mp
import time

def getVideoCapture(src=0):
    return cv.VideoCapture(src)


def main():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()  # Read Pose docs for parameters

    capture = getVideoCapture()
    pTime = 0

    while True:
        success, img = capture.read()

        results = pose.process(img)

        if results.pose_landmarks and results.pose_landmarks is not None:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f"FPS:{str(int(fps))}", (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("Video", img)

        # q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()