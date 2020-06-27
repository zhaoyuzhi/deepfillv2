import numpy as np
import cv2

# color scribble

if __name__ == "__main__":
    
    def color_scribble(img, color_point = 30, color_width = 5):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)

        times = np.random.randint(color_point)
        print(times)
        for i in range(times):
            # random selection
            rand_h = np.random.randint(height)
            rand_w = np.random.randint(width)
            # define min and max
            min_h = rand_h - (color_width - 1) // 2
            max_h = rand_h + (color_width - 1) // 2
            min_w = rand_w - (color_width - 1) // 2
            max_w = rand_w + (color_width - 1) // 2
            min_h = max(min_h, 0)
            min_w = max(min_w, 0)
            max_h = min(max_h, height)
            max_w = min(max_w, width)
            # attach color points
            scribble[min_h:max_h, min_w:max_w, :] = img[rand_h, rand_w, :]

        return scribble
    
    def blurish(img, color_blur_width = 11):
        img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
        return img
    
    img = cv2.imread('example.JPEG')
    scribble = color_scribble(img)
    scribble = blurish(scribble)

    show = np.concatenate((img, scribble), axis = 1)
    #show = cv2.resize(show, (show.shape[1] // 2, show.shape[0] // 2))
    cv2.imshow('scribble', show)
    cv2.waitKey(0)
