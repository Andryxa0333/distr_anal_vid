import cv2 as cv


video_input = cv.VideoCapture('/home/andr/antonov_proj/input/1.mp4')

if not video_input.isOpened():
    raise RuntimeError("Error, while opening video")

counter = 0
while video_input.isOpened():
    ret, frame = video_input.read()
    if not ret:
        break

    frame = cv.resize(frame,(640, int(frame.shape[0]/(frame.shape[1]/640))), interpolation=cv.INTER_AREA)

    counter+=1
    print(counter, frame.shape)
    # print(frame, "\n\n\n")

    cv.imwrite('test.png', frame)

def send_to_inf():
    pass

def get_from_inf():
    pass

