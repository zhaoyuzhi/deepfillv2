import numpy as np
import cv2
import matplotlib.pyplot as plt

# image completion
# cv2.copyMakeBoder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)

# cv2.BORDER_REPLICATE：    进行复制的补零操作，只对边缘的点进行复制，然后该列上的点都是这些
# cv2.BORDER_REFLECT:       进行翻转的补零操作，举例只对当前对应的边缘    gfedcba|abcdefgh|hgfedcb
# cv2.BORDER_REFLECT_101：  进行翻转的补零操作                          gfedcb|abcdefgh|gfedcb
# cv2.BORDER_WRAP:          进行上下边缘调换的外包复制操作               bcdegh|abcdefgh|abcdefg

if __name__ == "__main__":
    # read one image
    root = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\COCO2014_train_256\\COCO_train2014_000000000049.jpg'
    img = cv2.imread(root)
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    # REPLICATE: 复制最边缘上的一个点，所有的维度都使用当前的点
    REPLICATE = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
    # REFLECT: 进行翻转，即 gfedcba|abcdefgh|hgfedcb, 对于两侧的数据而言
    REFLECT = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
    # REFLECT_101: 进行按中间值翻转 gfedcb|abcdefgh|gfedcb
    REFLECT_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
    # WRAP: 外包装法   bcdefgh|abcdefgh|abcdefg， 相当于进行了上下的复制
    WRAP = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
    # CONST: 进行常数的补全操作, value = 0，表示使用0进行补全操作
    CONST = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value = 0)
    # plot
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    REPLICATE = cv2.cvtColor(REPLICATE, cv2.COLOR_BGR2RGB)
    REFLECT = cv2.cvtColor(REFLECT, cv2.COLOR_BGR2RGB)
    REFLECT_101 = cv2.cvtColor(REFLECT_101, cv2.COLOR_BGR2RGB)
    WRAP = cv2.cvtColor(WRAP, cv2.COLOR_BGR2RGB)
    CONST = cv2.cvtColor(CONST, cv2.COLOR_BGR2RGB)
    plt.subplot(231)
    plt.imshow(img), plt.title('ORIGINAL')
    plt.subplot(232)
    plt.imshow(REPLICATE), plt.title('REPLICATE')
    plt.subplot(233)
    plt.imshow(REFLECT), plt.title('REFLECT')
    plt.subplot(234)
    plt.imshow(REFLECT_101), plt.title('REFLECT_101')
    plt.subplot(235)
    plt.imshow(WRAP), plt.title('WRAP')
    plt.subplot(236)
    plt.imshow(CONST), plt.title('CONSTANT')
    plt.show()
