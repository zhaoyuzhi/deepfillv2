import numpy as np
import cv2

# image mask

# free form mask
# bbox mask

if __name__ == "__main__":
    # choice corresponds to mask type
    choice = 'ff'

    if choice == 'ff':

        config={'img_shape': [256, 256], 'mv': 15, 'ma': 4.0, 'ml': 40, 'mbw': 5}

        h, w = config['img_shape']
        mask = np.zeros((h,w))
        num_v = np.random.randint(config['mv'])

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(config['ma'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(config['ml'])
                brush_w = 5 + np.random.randint(config['mbw'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 255.0, brush_w)
                start_x, start_y = end_x, end_y

        mask = mask.astype(np.uint8)
        cv2.imshow("free form mask", mask)
        cv2.waitKey(0)

    if choice == 'bbox':

        shape = [256, 256]
        margin = [10, 10]
        bbox_shape = [30, 30]

        def random_bbox(shape, margin, bbox_shape):
            """Generate a random tlhw with configuration.
            Args:
                config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
            Returns:
                tuple: (top, left, height, width)
            """
            img_height, img_width = shape
            height, width = bbox_shape
            ver_margin, hor_margin = margin
            maxt = img_height - ver_margin - height
            maxl = img_width - hor_margin - width
            t = np.random.randint(low = ver_margin, high = maxt)
            l = np.random.randint(low = hor_margin, high = maxl)
            h = height
            w = width
            return (t, l, h, w)

        bboxs = []
        for i in range(20):
            bbox = random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)

        height, width = shape
        mask = np.zeros((height, width), np.float32)
        #print(mask.shape)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 255.

        mask = mask.astype(np.uint8)
        cv2.imshow("free form mask", mask)
        cv2.waitKey(0)
