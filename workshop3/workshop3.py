import numpy as np
import cv2.cv2
import matplotlib.pyplot as plt


def orb_stitcher(imgs):
    # find the keypoints with ORB
    orb1 = cv2.ORB_create(1000,1.1,13)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp_master, des_master = orb1.detectAndCompute(imgs[0], None)
    kp_secondary, des_secondary = orb1.detectAndCompute(imgs[1], None)


    matches = bf.match(des_secondary,des_master )
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    selected = []
    for m in matches:
        if m.distance< 40:
            selected.append(m)

    out_img = cv2.drawMatches(imgs[1], kp_secondary, imgs[0], kp_master,  selected,None)
    cv2.namedWindow('www', cv2.WINDOW_NORMAL)
    cv2.imshow('www',out_img)
    # cv2.imwrite('matches.jpg',out_img)
    cv2.waitKey(0)

    warped = None
    if len(selected)>10:

        dst_pts = np.float32([kp_master[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_secondary[m.queryIdx].pt for m in selected]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts,dst_pts,  cv2.RANSAC, 5.0)


        h, w = imgs[0].shape[0:2]
        pts = np.float32([[0, 0],[w,0], [w, h], [0, h], [0, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        max_extent = np.max(dst,axis = 0)[0].astype(np.int)[::-1]
        sz_out = (max(max_extent[1],imgs[0].shape[1]),max(max_extent[0],imgs[0].shape[0]))

        # img2 = cv2.polylines(imgs[0], [np.int32(dst)], True, [0,255,0], 3, cv2.LINE_AA)

        cv2.namedWindow('w',cv2.WINDOW_NORMAL)


        warped = cv2.warpPerspective(imgs[1],M,dsize=sz_out)
        img_for_show = warped.copy()
        img_for_show[0:imgs[0].shape[0],0:imgs[0].shape[1],1] =  imgs[0][:,:,1]
        cv2.imshow('w', img_for_show)
        cv2.waitKey(0)
    return warped

def allign_sizes(imgs):
    mx,my = 0,0
    for im in imgs:
        mx = max(mx,im.shape[0])
        my = max(my,im.shape[1])
    out_imgs = []
    for im in imgs:
        single_out = np.zeros((mx,my,3),dtype=np.uint8)
        single_out[0:im.shape[0],0:im.shape[1],:]= im
        out_imgs.append(single_out)
    return out_imgs



def stitching_demo():

    imgs = []
    imgs.append(cv2.imread('img0.jpg'))
    imgs.append(cv2.imread('img1.jpg'))
    imgs.append(cv2.imread('img2.jpg'))
    warp1 = orb_stitcher(imgs[:2])
    warp2 = orb_stitcher([imgs[0],imgs[2]])
    imgs_out = allign_sizes([imgs[0],warp1,warp2])

    single_out = np.zeros_like(imgs_out[0])
    for i in range(3):
        single_out[:,:,i] = imgs_out[i][:,:,i]

    cv2.namedWindow('final_warp',cv2.WINDOW_NORMAL)
    cv2.imshow('final_warp',single_out)
    cv2.waitKey(0)
    cv2.imwrite('stitched.jpg',single_out)



def of_demo():
    pixels_cut = 50
    pixels_cut_left = 100

    cap = cv2.VideoCapture('rally.avi')
    # cap = cv2.VideoCapture('input.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.2,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(35, 35),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (1000, 3))

    # Take first frame and find corners in it

    ret, old_frame = cap.read()
    old_frame = old_frame[:-pixels_cut, pixels_cut_left:, :]
    out = cv2.VideoWriter('output2.avi', fourcc, 30.0, (old_frame.shape[1], old_frame.shape[0]))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    frno = 0
    restart = False
    while (1):
        frno += 1
        ret, frame = cap.read()
        frame = frame[:-pixels_cut, pixels_cut_left:, :]
        if ret and frno < 70:

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if restart:
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                restart = False
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            successful = (st == 1)
            if np.sum(successful) == 0:
                restart = True
            # Select good points
            good_new = p1[successful]
            good_old = p0[successful]

            # draw the tracks
            count_of_moved = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                velocity = np.sqrt((a - c) ** 2 + (b - d) ** 2)
                if velocity > 1:
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 4, color[i].tolist(), -1)
                    count_of_moved += 1


            # mask_of_mask = cv2.inRange(mask, (0, 0, 0), (3, 3, 3))/255
            # frame = frame*(np.expand_dims(mask_of_mask.astype(np.uint8),axis=2))
            img = cv2.add(frame, mask)

            mask = np.round(mask.astype(np.float) / 1.1).astype(np.uint8)

            cv2.imshow('frame', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            out.write(img)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


def dense_of_demo():

    def draw_flow(img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(np.int32)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def draw_hsv(flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 15, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

    cam = cv2.VideoCapture('rally.avi')


    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    out = cv2.VideoWriter('farn.avi', fourcc, 30.0, (prevgray.shape[1], prevgray.shape[0]))

    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.write(img)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 1, 0.5, 5, 15, 3, 1, 1.2, flags=1)
        prevgray = gray

        cv2.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            hsv_img = draw_hsv(flow)
            cv2.imshow('flow HSV', hsv_img)
            out.write(hsv_img)
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)
            out.write(cur_glitch)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv

        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    of_demo()
    # dense_of_demo()
    # stitching_demo()