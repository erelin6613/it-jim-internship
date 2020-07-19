import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    solution = 1
    cap = cv2.VideoCapture('input_video.avi')
    frame_count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = frame[60:-60, :, :]
            frm = frame
            if solution == 1:
                # Making a copy of original frame and conversion to LAB
                processed_frm = frm.copy()
                processed_frm = cv2.cvtColor(processed_frm, cv2.COLOR_BGR2LAB)

                # Making a yellow mask
                yellow_mask = processed_frm[:, :, 2]
                yellow_mask = cv2.medianBlur(yellow_mask, 7)
                ret, yellow_mask = cv2.threshold(yellow_mask, 140, 255, cv2.THRESH_BINARY)  # good for non-equalized colors
                kernel = np.ones((17, 17), np.uint8)
                yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)

                # Making a green mask
                green_channel = 255 - processed_frm[:, :, 1]
                yellow_channel = processed_frm[:, :, 2]
                light_green_mask = np.uint8((1.6 * green_channel[:, :] + 0.4 * yellow_channel[:, :]) / 2)
                light_green_mask = cv2.medianBlur(light_green_mask, 7)
                ret, light_green_mask = cv2.threshold(light_green_mask, 135, 255, cv2.THRESH_BINARY)  # good for non-equalized colors
                kernel = np.ones((11, 11), np.uint8)
                light_green_mask = cv2.erode(light_green_mask, kernel, iterations=1)

                # Making another copy of original frame, equalization of each color channel and conversion to LAB
                processed_frm = frm.copy()
                for i in range(3):
                    processed_frm[:, :, i] = cv2.equalizeHist(processed_frm[:, :, i])
                processed_frm = cv2.cvtColor(processed_frm, cv2.COLOR_BGR2LAB)

                # Making a green mask
                magenta_mask = processed_frm[:, :, 1]
                magenta_mask = cv2.medianBlur(magenta_mask, 11)
                ret, magenta_mask = cv2.threshold(magenta_mask, 185, 255, cv2.THRESH_BINARY)  # good for equalized colors

                # Making a copy of original frame and conversion to grayscale
                processed_frm = frm.copy()
                processed_frm = cv2.cvtColor(processed_frm, cv2.COLOR_BGR2GRAY)

                # Making a black mask
                black_mask = 255 - processed_frm
                ret, black_mask = cv2.threshold(black_mask, 230, 255, cv2.THRESH_BINARY)  # good for equalized colors
                kernel = np.ones((12, 12), np.uint8)
                black_mask = cv2.dilate(black_mask, kernel, iterations=1)

                # Masks binarization to apply Boolean OR operation
                yellow_mask = yellow_mask / 255
                light_green_mask = light_green_mask / 255
                magenta_mask = magenta_mask / 255
                black_mask = black_mask / 255

                # Making result binary mask
                result_mask = np.logical_or(yellow_mask, light_green_mask)
                result_mask = np.logical_or(result_mask, magenta_mask)
                result_mask = np.logical_or(result_mask, black_mask)

                # Making binary mask suitable for OpenCV showing and saving
                result_mask = np.uint8(result_mask * 255)

                # Making the grey scale image have three channels
                threshed = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2BGR)

            if solution == 2:
                # convert to hsv
                frm_hsv = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
                frm_split_ch = np.zeros((frm_hsv.shape[0], frm_hsv.shape[1] * frm_hsv.shape[2]), dtype=np.uint8)
                # filling in the empty image by channels
                for i in range(frm_hsv.shape[2]):
                    frm_split_ch[:, frm_hsv.shape[1] * i:frm_hsv.shape[1] * (i + 1)] = frm_hsv[:, :, i]

                # use gaussian blur to get rid of random noise
                blur = cv2.GaussianBlur(frm_split_ch, (5, 5), 3)

                # apply CLAHE method to make the shapes stand out
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
                cl = clahe.apply(blur)

                # mask for green and yellow
                mask_gy = cv2.inRange(frm_hsv, np.array([20, 40, 100]), np.array([80, 100, 255]))

                # mask for pink
                mask_p = cv2.inRange(frm_hsv, np.array([140, 50, 0]), np.array([179, 150, 255]))

                # mask for black
                mask_b = cv2.inRange(frm_hsv, np.array([0, 0, 0]), np.array([179, 255, 40]))

                threshed = mask_gy + mask_p + mask_b
                threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

            if solution == 3:
                threshed = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ret, threshed0 = cv2.threshold(threshed[:,:,0],50,255,cv2.THRESH_BINARY)
                ret, threshed1 = cv2.threshold(threshed[:,:,1],150,255,cv2.THRESH_BINARY)
                ret, threshed2 = cv2.threshold(threshed[:,:,2],120,255,cv2.THRESH_BINARY)
                threshed = 255 - np.minimum(np.minimum(threshed0, 255 - threshed1),threshed2)

                threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

            if solution == 4:
                frm0, frm1, frm2 = cv2.split(frm)

                frm0 = cv2.blur(frm0, (11, 11))

                frm0 = cv2.normalize(frm0, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                frm0 = cv2.medianBlur(frm0, ksize=11)

                # пытаемся получить адаптивным порогом сразу хороший результат, без контуров
                frm0 = cv2.adaptiveThreshold(frm0, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 131, 20)

                frm0 = cv2.erode(frm0, kernel=np.ones((13, 13), dtype=np.uint8))
                frm0 = cv2.dilate(frm0, kernel=np.ones((13, 13), dtype=np.uint8))

                # инвертируем, чтобы маска была белой, а фон черным
                threshed = cv2.bitwise_not(frm0)
                threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

            threshed = threshed[:,:,1]
            #threshed = np.hstack((frm, threshed))
            #threshed = np.minimum(frm, threshed)
            cv2.imshow('Video frame', threshed)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break



if __name__ == '__main__':
    main()