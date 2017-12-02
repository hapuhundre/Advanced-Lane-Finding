import numpy as np
import cv2

def Drawing(undist, warped, leftx, rightx, ploty, curve_rad, position):
    # 创建一个空图
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])

    # 将x和y点重新转换为cv2.fillPoly() 的可用格式
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # 将车道绘制到变形的空白图像上
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # 使用反向透视矩阵（Minv）将空白图像逆转至原始图像空间
    src = np.float32([[490, 470],[810, 470],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                      [1250, 720],[40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp,Minv,(undist.shape[1], undist.shape[0]))
    
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int(curve_rad)), (100,80),
                fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    
    # 900 is based on the width of the lane (in pixels) at the bottom of birds eye view images
    # 鸟俯视图中底部的宽度
    dist = abs((640 - position)*3.7/900)
    
    if position > 640:
        msg = 'Vehicle is {:.2f}m left of center'.format(dist)
    else:
        msg = 'Vehicle is {:.2f}m right of center'.format(dist)
    cv2.putText(result, msg, (120,140), fontFace = 16, fontScale = 2, 
                color=(255,255,255), thickness = 2)
    return result

