import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def Undistort(img):
    # 记载相机内参
    dist_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb") )
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    # 恢复图片扭曲
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img

def PerspectiveTransform(img):
    undist_img = Undistort(img)
    m,n = img.shape[0], img.shape[1] # 
    img_size = (n, m) # 这里注意一下是反的
    # from proj 1 of term 1, oh a funny journey....
    # src = np.float32([[490, 482],[810, 482],
    #                   [1250, 720],[40, 720]])
    # dst = np.float32([[0, 0], [1280, 0], 
    #                  [1250, 720],[40, 720]])
    src = np.float32([[490, 470],[810, 470],
                      [1280, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1280, 720],[40, 720]])
    ##---
    # 透视变换，又称鸟视。。。see car lane using bird eye ^o^
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

#--------------------------test--------------------------------------------------

def plot(img, imged, msg = "Changed Image"):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(cv2.cvtColor(imged, cv2.COLOR_BGR2RGB))
    ax2.set_title(msg, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def test1():
    """
    图片扭曲测试
    """
    img = cv2.imread('test_images/test1.jpg')
    undist = Undistort(img)
    plot(img, undist, msg = 'Undistort Image')

def test2():
    img = cv2.imread('test_images/test1.jpg')
    undist = PerspectiveTransform(img)
    plot(img, undist, msg = 'Transformed Image')
#---------------------------------------------------------------------------

if __name__ == '__main__':
    test2()


