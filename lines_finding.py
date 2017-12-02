import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LinesFinding:
    def __init__(self):
        # 是否使用过窗口搜索
        self.is_found = False
        # 上一图像搜索得到的左右拟合曲线参数
        self.left_fit = None
        self.right_fit = None
        # 车道线曲率半径与位置
        self.curve_rad = 0
        self.position = 0

        # 在绘制行车区域会用到的属性
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None

    def curverad(self, img):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fit = self.left_fit
        right_fit = self.right_fit

        # NOTICE: We've calculated the radius of curvature based on pixel values, 
        # so the radius we are reporting is in pixel space, 
        # which is not the same as real world space. 
        # So we actually need to repeat this calculation 
        # after converting our x and y values to real world space.
        # 注意要考虑图片和像素是两个空间
        ym_per_pix = 30/720 # y方向上每像素的实际长度
        xm_per_pix = 3.7/700
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        # 拟合实际空间上的多项式曲线
        left_fit_cr = np.polyfit(ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        self.curve_rad = (left_curverad + right_curverad) / 2
        
        left_lane = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
        right_lane = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
        
        self.position = (right_lane + left_lane)/2

    def window_search(self, bin_warped):
        # 图像下半部分的直方图
        hist = np.sum(bin_warped[bin_warped.shape[0]//2:,:], axis=0)
        # 创建一个输出图像来绘制和可视化结果
        out_img = np.dstack((bin_warped, bin_warped, bin_warped))*255
        # 找出直方图左右两半的高峰
        # 这些将成为左右线的起点
        midpoint = np.int(hist.shape[0]/2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint
    
        # 滑移窗口的个数
        nwindows = 9
        # 窗口高度
        window_height = np.int(bin_warped.shape[0]/nwindows)
        # 确定图像中所有非零像素的x和y位置
        nonzero = bin_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # 窗口滑移当前位置
        curr_leftx = leftx_base
        curr_rightx = rightx_base
        # 窗口宽度
        margin = 100
        # 设置找到最近的像素的最小数量
        minpix = 50
        # 创建空列表以接收左右车道像素的索引
        left_lane_inds = []
        right_lane_inds = []
    
        for window in range(nwindows):
            # 确定窗口左右边界
            win_y_low = bin_warped.shape[0] - (window+1)*window_height
            win_y_high = bin_warped.shape[0] - window*window_height
            win_xleft_low = curr_leftx - margin
            win_xleft_high = curr_leftx + margin
            win_xright_low = curr_rightx - margin
            win_xright_high = curr_rightx + margin
            # 在图像上添加窗口
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),\
            #               (win_xleft_high,win_y_high),(0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),\
            #               (win_xright_high,win_y_high),(0,255,0), 2)
            # 确定窗口内x和y方向上的非零像素
            nozero_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            nozero_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &  
                                 (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # 添加至索引列表中
            left_lane_inds.append(nozero_left_inds)
            right_lane_inds.append(nozero_right_inds)
            # 如果发现的像素点大于minpix，则在其平均位置上重新调用下一个窗口
            if len(nozero_left_inds) > minpix:
                curr_leftx = np.int(np.mean(nonzerox[nozero_left_inds]))
            if len(nozero_right_inds) > minpix:        
                curr_rightx = np.int(np.mean(nonzerox[nozero_right_inds]))
        
        # 连接索引数组
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # 提取像素点位置
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # 多项式插值
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)


        self.is_found = True

        self.ploty = np.linspace(0, bin_warped.shape[0]-1, bin_warped.shape[0])
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        self.curverad(bin_warped)

    def direct_search(self, img):
        """
        在前一帧得到的多项式拟合基础上加一个边界(margin)
        边界内的点作为待拟合的点
        """
        left_fit = self.left_fit
        right_fit = self.right_fit

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                          left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                          left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                          right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                          right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        

        # Fit a second order polynomial to each
        # 如果图像上没有点的情况
        if len(leftx)<3 or len(lefty)<3 or len(rightx)<3 or len(righty)<3:
            self.window_search(img)
        else:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        
        self.ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        self.curverad(img)

    def visual(self, img):
        """
        可视化拟合曲线
        """
        if self.left_fit is None:
            print("Error! Cannot display the line without window search.")
            return 0
        left_fit = self.left_fit
        right_fit = self.right_fit
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image', fontsize=15)

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # out_img = np.copy(img)
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        ax2.imshow(img, cmap='gray')
        ax2.plot(left_fitx, ploty, color='green')
        ax2.plot(right_fitx, ploty, color='green')

        ax2.set_title('Poly Image', fontsize=15)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()   

if __name__ == '__main__':
    file_name = 'test_images/lines_finding_test.jpg'
    lines = LinesFinding()
    test_img = mpimg.imread(file_name)
    lines.window_search(test_img)
    lines.visual(test_img)
    lines.curverad(test_img)