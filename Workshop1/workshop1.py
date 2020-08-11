import cv2
import numpy as np



def run_sample_functions():
    '''
    Simple function which shows some of image manipulation routines
    :return: No return
    '''

    # Opening of an image
    image_filename = 'img_example.jpg'
    img_bgr = cv2.imread(image_filename)
    # The image is opened as Blue-Green-Red (opposite to RGB)


    # Resize of the image
    img_resized = cv2.resize(img_bgr,(img_bgr.shape[1]//6,img_bgr.shape[0]//6)) 
    # height - img_bgr.shape[1], width - img_bgr.shape[0]
    # Note that the shapes of the image in opencv and in numpy are in reverse order   (img.shape[1], img.shape[0])


    # Showing images
    cv2.imshow('Window with example', img_resized)
    cv2.waitKey(0)
    # won't draw anything without this function. Argument - time in milliseconds, 0 - until key pressed

    # convesion of the image into another colorspace
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)


    show_img_by_channels(img_hsv)


def show_img_by_channels(img_hsv):
    '''Function which draws all channels as a wider single-channel image.'''

    # empty image for results
    img_split_ch = np.zeros((img_hsv.shape[0],img_hsv.shape[1]*img_hsv.shape[2]),dtype=np.uint8)
    #filling in the empty image by channels
    for i in range(img_hsv.shape[2]):
        img_split_ch[:,img_hsv.shape[1]*i:img_hsv.shape[1]*(i+1)] = img_hsv[:,:,i]
    cv2.imshow('Split channel image',img_split_ch)
    cv2.waitKey(0)


def run_video_manipulation_samples():
    video_name = 'sample.avi'
    cap = cv2.VideoCapture(video_name)
    ret,frm = cap.read()
    frm_count = 0
    key = None #
    while ret:
        #frm = cv2.Sobel(frm, cv2.CV_64F, 0, 1, ksize=5) # Sobel
        # frm = cv2.Laplacian(frm, cv2.CV_64F)#, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
        # frm = cv2.dilate(frm, kernel=np.ones((3, 3), dtype=np.uint8))
        frm = cv2.Canny(frm, 33, 150)
        cv2.putText(frm,'frame number: ' + str(frm_count),(100,100),0,2,[255,255,255],5)
        cv2.imshow('Video frame',frm)

        # Pause on pressing of space.
        if key == ord(' '):
            wait_period = 0
        else:
            wait_period = 30

        #drawing, waiting, getting key, reading another frame
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()

    return 0

def run_video_crop():
    '''Simple function for cropping of a video and saving into a new file'''
    video_name = 'large_sample.mp4'
    cap = cv2.VideoCapture(video_name)
    ret,frm = cap.read()
    frm_count = 0

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Setting up new video writer
    frames_per_second = 30
    image_size = (frm.shape[1],frm.shape[0])
    writer = cv2.VideoWriter('sample.avi', fourcc, frames_per_second,image_size )

    # reading, writing and showing
    while ret and frm_count<50:
        writer.write(frm)
        cv2.imshow('Video frame',frm)
        cv2.waitKey(10)
        ret, frm = cap.read()
        frm_count+=1

    # don't forget to release the writer. Otherwise, the file may be corrupted.
    cap.release()
    writer.release()

    return 0

if __name__ == '__main__':
    run_video_manipulation_samples()
    # run_sample_functions()