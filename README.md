一个简单的数字水印程序

main.py内是自己写的，基于傅里叶变换的数字水印，效果不是特别的好。可能是水印本身信息太少的原因，也有可能是方法的原因。
test.py内是一个第三方的库函数：blind-watermark，效果不错，但限制条件比较多，比如对水印的的大小和原图的大小，解水印需要水印的size等。