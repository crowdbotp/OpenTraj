import tkinter as tk
import cv2
import numpy as np
import argparse


def click_src(event, x, y, flags, param):
    global update_flag, points_src
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_src) > len(points_dst):
            print('Error! set the corresponding point in dst image')
            return
        points_src.append(np.array([x, y]))
        update_flag = True


def click_dst(event, x, y, flags, param):
    global update_flag, points_dst
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_dst) > len(points_src):
            print('Error! set the corresponding point in src image')
            return
        points_dst.append(np.array([x, y]))
        update_flag = True
    if event == cv2.EVENT_RBUTTONDOWN:
        pass  # remove point


def update():
    global im_src, im_dst, im_src_warped, im_src_orig, im_dst_orig, update_flag
    if update_flag:
        im_src = im_src_orig.copy()
        im_dst = im_dst_orig.copy()
        for pnt in points_src:
            cv2.circle(im_src, (pnt[0], pnt[1]), 3, (255, 0, 0), 2)
        for pnt in points_dst:
            cv2.circle(im_dst, (pnt[0], pnt[1]), 3, (0, 255, 0), 2)

        H = np.eye(3)
        if len(points_src) > 3 and len(points_dst) > 3:
            n_pnts = min(len(points_src), len(points_dst))
            points_src_np = np.array(points_src[:n_pnts]).astype(np.float64)
            points_dst_np = np.array(points_dst[:n_pnts]).astype(np.float64)
            H, _ = cv2.findHomography(points_src_np, points_dst_np)
            print('Homography = \n', H)
        im_src_warped = cv2.warpPerspective(im_src, H, (im_dst.shape[1], im_dst.shape[0]))
        dst_scale = float(dst_scale_wgt.get())

        H_out = H.copy()
        H_out[0, 0] = H[1, 1] * dst_scale
        H_out[0, 1] = H[1, 0] * dst_scale
        H_out[0, 2] = H[1, 2] * dst_scale
        H_out[1, 0] = H[0, 1] * dst_scale
        H_out[1, 1] = H[0, 0] * dst_scale
        H_out[1, 2] = H[0, 2] * dst_scale
        H_out[2, 0] = H[2, 1]
        H_out[2, 1] = H[2, 0]
        H_out[2, 2] = H[2, 2]

        txt.delete('1.0', tk.END)
        txt.insert(tk.END, str(H_out))

        update_flag = False

    cv2.imshow("src", im_src)
    cv2.imshow("dst", im_dst)

    mix = cv2.addWeighted(im_dst, 0.5, im_src_warped, 0.5, 0)
    cv2.imshow("mix", mix)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        exit(1)
    # elif key == 13 and len(points_src) == len(points_dst):

    window.after(50, update)


def set_update_flag():
    global update_flag
    update_flag = True


def reset_points():
    global points_src, points_dst, update_flag
    points_src, points_dst = [], []
    update_flag = True


'''
   Usage Example: 
   cd OpenTraj/tools
   find_homography.py --src '../GC/reference.jpg' --dst '../GC/plan.png' --dst-scale 0.06
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='', type=str)
    parser.add_argument('--dst', default='', type=str)
    parser.add_argument('--dst-scale', default=1., type=float, help='scale of dst image')
    args = parser.parse_args()

    im_src_orig = cv2.imread(args.src)
    im_dst_orig = cv2.imread(args.dst)

    im_src = im_src_orig.copy()
    im_dst = im_dst_orig.copy()
    im_src_warped = im_dst_orig.copy()
    im_mix = np.zeros_like(im_dst)
    dst_scale = args.dst_scale
    points_src = []
    points_dst = []
    update_flag = True

    window = tk.Tk()
    window.title("Parameters")
    window.minsize(200, 300)

    lbl_1 = tk.Label(window, text="Scale:")
    lbl_1.grid(column=0, row=0)

    dst_scale_wgt = tk.Entry(window)
    dst_scale_wgt.insert(0, "1.0")
    dst_scale_wgt.grid(column=1, row=0)

    btn_update = tk.Button(window, text="Update", command=set_update_flag)
    btn_update.grid(column=2, row=0)

    lbl_2 = tk.Label(window, text="Homography:")
    lbl_2.grid(column=0, row=1)

    txt = tk.Text(window, height=10, width=50)
    txt.insert(tk.END, "-1")
    txt.grid(column=0, row=2, columnspan=3)

    btn_reset = tk.Button(window, text="Reset Points", command=reset_points)
    btn_reset.grid(column=0, row=3, columnspan=3)

    cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mix", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("src", click_src)
    cv2.setMouseCallback("dst", click_dst)
    cv2.setMouseCallback("mix", click_dst)

    window.after(50, update)
    window.mainloop()

