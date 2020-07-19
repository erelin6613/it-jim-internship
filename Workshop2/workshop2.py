import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_contours_demo():
    image_filename = 'contours_example.jpg'
    img_bgr = cv2.imread(image_filename)
    img_thresh = cv2.adaptiveThreshold(img_bgr[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,5)
    img_thresh = cv2.dilate(img_thresh,np.ones((3,3)))
    img_thresh = cv2.erode(img_thresh, np.ones((11,11)))

    conts, hier = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in conts:
        clr = np.random.randint(100,200,(3,),dtype=np.uint8)

        x,y,w,h = cv2.boundingRect(cnt)
        if w>10 and h>10:
            cv2.circle(img_bgr,(x+w//2,y+h//2),10,[255,0,255],5)
            cv2.drawContours(img_bgr,[cnt],-1,clr.tolist(),5)

            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img_bgr,(cx,cy),10,[255,0,0],3)

    cv2.namedWindow('win',cv2.WINDOW_NORMAL)
    cv2.imshow('win',img_bgr)
    cv2.waitKey(0)


def run_template_match_demo():
    '''
    Playing with template matcher
    :return: No return
    '''

    # Opening of an image
    image_filename = 'img_example.jpg'
    img_bgr = cv2.imread(image_filename)
    template_filename = 'template.png'
    tm_bgr = cv2.imread(template_filename)

    matching = cv2.matchTemplate(img_bgr,tm_bgr,cv2.TM_CCOEFF_NORMED)
    matching = cv2.normalize(matching,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    # matching = cv2.resize(matching, (matching.shape[1] // 3, matching.shape[0] // 3))
    plt.imshow(matching)
    plt.show()
    thresh = cv2.threshold(matching,np.max(matching)*0.6,255,cv2.THRESH_BINARY)[1]
    conts,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_draw = img_bgr.copy()

    cv2.drawContours(img_draw,conts,-1,[0,255,0],3)
    cv2.namedWindow('win',cv2.WINDOW_NORMAL)
    cv2.imshow('win',img_draw)
    cv2.waitKey(0)

    for cnt in conts:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr,(x+w//2,y+h//2),(x+w//2+tm_bgr.shape[1],y+h//2+tm_bgr.shape[0]),[0,0,255],6)
        # cv2.drawContours( img_bgr,conts,-1,[0,0,255],15)

    img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // 3, img_bgr.shape[0] // 3))

    # Showing images
    cv2.imshow('Window with example', img_bgr)
    cv2.waitKey(0) # won't draw anything without this function. Argument - time in milliseconds, 0 - until key pressed


def running_win_hist_match(img,tmp):
    '''
    This funciton will split image into a mesh of patches and will perform hist match for each patch
    :param img: image to match
    :param tmp: template
    :return: 2d array with matching results
    '''
    def normed_hist(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [5, 5, 5],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    tmp_hist = normed_hist(tmp)

    num_splits_x, num_splits_y = 100, 70
    step_x = img.shape[1]//num_splits_x
    step_y = img.shape[0]//num_splits_y
    out_arr = np.zeros((num_splits_y,num_splits_x))
    for i in range(num_splits_x):
        for k in range(num_splits_y):
            patch = img[step_y*k:step_y*(k+1),step_x*i:step_x*(i+1),:]
            test_hist = normed_hist(patch)
            match = cv2.compareHist(tmp_hist,test_hist,cv2.HISTCMP_HELLINGER)
            out_arr[k,i] = match
    return out_arr


def run_hist_match_demo():
    '''
        Playing with template matcher
        :return: No return
        '''

    # Opening of an image
    image_filename = 'img_example.jpg'
    img_bgr = cv2.imread(image_filename)
    template_filename = 'template.png'
    tm_bgr = cv2.imread(template_filename)

    hist = cv2.calcHist([tm_bgr], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    plt.plot(hist)
    plt.show()

    map = running_win_hist_match(img_bgr,tm_bgr)


    plt.subplot(121)
    plt.imshow(map)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(img_bgr[:,:,::-1])
    plt.show()


if __name__ == '__main__':
    # run_hist_match_demo()
    # run_contours_demo()
    run_template_match_demo()


