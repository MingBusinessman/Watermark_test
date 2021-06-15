import cv2
from blind_watermark import WaterMark

logo_path = 'logo.png'
watermark_path = 'erweima_small.png'
output_path = 'logo_with_mark.png'

# erweima = cv2.imread('erweima.png', 0)
# erweima = cv2.resize(erweima, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# cv2.imwrite('erweima_small.png', erweima)
#
# watermark_path = 'erweima_small.png'

mission1 = WaterMark(password_wm=1, password_img=1)
mission1.read_img(logo_path)
mission1.read_wm(watermark_path)
mission1.embed(output_path)
mission1.extract(output_path, wm_shape=(64, 64), out_wm_name='extracted.png')

