import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from imgs_transf import PerspectiveTransform

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 计算方向梯度
    # Apply threshold
    # 灰阶转换
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV的Sobel()函数并取绝对值
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel_img = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and 应用阈值
    binary_img = np.zeros_like(scaled_sobel_img)
    binary_img[(scaled_sobel_img >= thresh[0]) & (scaled_sobel_img <= thresh[1])] = 1

    return binary_img

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 转换为灰阶
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 求x与y方向上的Sobel梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 计算梯度大小
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 重新调整到8位
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # 创建满足阈值的二进制图像，否则为零
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # 返回二进制图像
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def color_threshold(img, s_thresh=(170, 255), \
                    l_thresh=(170, 255), b_thresh=(170, 255)):
    """
    三种不同颜色空间的阈值
    HLS: 色相hue, 饱和度saturation,亮度lightness/luminance
    LUV: 
    Lab: L像素的亮度 a红色到绿色的范围 b黄色到蓝色的范围
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    s_channel = hls[:,:,2]
    l_channel = luv[:,:,0]
    b_channel = lab[:,:,2]

    s_bin = np.zeros_like(s_channel)
    s_bin[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    l_bin = np.zeros_like(l_channel)
    l_bin[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    b_bin = np.zeros_like(b_channel)
    b_bin[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    return s_bin, l_bin, b_bin

def Processing(img):
    """
    调整函数参数，对Sobel梯度与颜色阈值进行组合
    """
    s_bin, l_bin, b_bin = color_threshold(img, s_thresh=(180, 255), \
                          l_thresh=(225, 255), b_thresh=(155, 200))
    combined_bin = np.zeros_like(s_bin)
    combined_bin[(l_bin == 1) | (b_bin == 1)] = 1
    return combined_bin
#--------------test-----------------------
def plot(image, grad_binary, msg = "Changed Image"):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title(msg, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def test1(img):
    """
    sobel过滤测试
    无法提取黄线特征
    """
    # thresh = abs_sobel_thresh(img, orient='x', thresh=(20,100))
    # thresh = abs_sobel_thresh(img, orient='y', thresh=(20,100))
    # thresh = mag_thresh(img,mag_thresh=(40,150))
    #thresh = dir_threshold(img, thresh=(0.5, np.pi/2))
    plot(img, thresh)

def test2(img):
    """
    颜色过滤测试
    在泛白路（水泥路）适应性差
    """
    imged = color_threshold(img, channel_name='s', thresh=(170,255))
    plot(img, imged)

def test3(img):
    """
    pipeline 测试
    """
    piped = Processing(img)
    plot(img, piped)
    
#-----------------------------------------------------------------------

if __name__ == '__main__':
    """ 测试文件名
    straight_lines1~2.jpg
    test1~6.jpg
    """
    img = mpimg.imread('test_images/test3.jpg')
    trans = PerspectiveTransform(img)
    test3(trans)