import random
import cv2
import numpy as np
import os


def encode(img_path, wm_path, res_path, alpha):
    #读取源图像和水印图像
    img = cv2.imread(img_path)
    height, width, channel = np.shape(img)
    watermark = cv2.imread(wm_path)
    wm_height, wm_width = watermark.shape[0], watermark.shape[1]
    #源图像傅里叶变换
    img_f = np.fft.fft2(img)
    #水印图像编码
    #random
    x, y = list(range(int(height / 2))), list(range(width))
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)  # 与源图像等大小的模板，用于加上水印
    for i in range(int(height / 2)):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = watermark[x[i]][y[j]]
                tmp[height - 1 - i][width - 1 - j] = tmp[i][j]
    #混杂
    res_f = img_f + alpha * tmp
    #逆变换
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def decode(ori_path, img_path, res_path, alpha):
    ori = cv2.imread(ori_path)
    img = cv2.imread(img_path)
    height, width = ori.shape[0], ori.shape[1]
    #源图像与水印图像傅里叶变换
    ori_f = np.fft.fft2(ori)
    img_f = np.fft.fft2(img)
    watermark = (ori_f - img_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(ori.shape)

    #获取随机种子
    random.seed(height + width)
    x = list(range(int(height / 2)))
    y = list(range(width))
    random.shuffle(x)
    random.shuffle(y)
    for i in range(int(height / 2)):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':

    #嵌入水印
    logo_path = 'logo.png'
    watermark_path = 'watermark_enlarged.png'
    output_path = 'logo_with_mark.png'
    alpha = 5
    encode(logo_path, watermark_path, output_path, alpha)

    #解水印
    decode_res_path = 'decode_res.png'
    decode(logo_path, output_path, decode_res_path, alpha)

    # mission1 = WaterMark(password_wm=1, password_img=1)
    # mission1.read_img(logo_path)
    # mission1.read_wm(watermark_path)
    # mission1.embed(output_path)
    #
    #