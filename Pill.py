import cv2 as cv
import numpy as np
import imutils

class Pill:

    """Class that encapsulates all the methods and variables needed to extract a pll from the image"""

    def __init__(self, path):
        self.image = cv.imread(path)
        self.path = path
        self.contour = None
        self.__process_image()


    def __get_contours(self, mask, limit=0):
        cnts, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if (limit!= 0):
            cnts = [cnt for cnt in cnts if cv.contourArea(cnt) > limit]
        
        return cnts
    

    def __grab_cut(self, mask):
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        mask[mask > 0] = cv.GC_PR_FGD
        mask[mask == 0] = cv.GC_BGD
        (mask, bgModel, fgModel) = cv.grabCut(self.image, mask, None, bgModel,fgModel, iterCount=6, mode=cv.GC_INIT_WITH_MASK)

        values = (
            ("Definite Background", cv.GC_BGD),
            ("Probable Background", cv.GC_PR_BGD),
            ("Definite Foreground", cv.GC_FGD),
            ("Probable Foreground", cv.GC_PR_FGD),
        )
        valueMask = (mask == values[3][1]).astype("uint8") * 255
    
        return valueMask


    def __process_image(self):
        hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        blurred = cv.GaussianBlur(hsv,(3,3),0)
        mask = cv.inRange(blurred, (1,0, 0), (179, 255, 255))

        eroded = cv.erode(mask, None, iterations=4)
        dilated = cv.dilate(eroded, None, iterations=4)

        grabbed = self.__grab_cut(dilated)
        self.contour = self.__get_contours(grabbed, 500)
        self.image = cv.bitwise_and(self.image, self.image, mask=grabbed)


    def resize(self, scale):
        width = int(self.image.shape[1] * scale / 100)
        height = int(self.image.shape[0] * scale / 100)
        dim = (width, height)

        # resize image
        self.image = cv.resize(self.image, dim, cv.INTER_AREA)
        
        if self.contour is None:
            raise Exception('Image has not been processed yet')

        self.contour = cv.resize(self.contour, dim, cv.INTER_AREA)


    def rotate(self, degree):
        self.image = imutils.rotate_bound(self.image, degree)

        if self.contour is None:
            raise Exception('Image has not been processed yet')
        
        self.contour = imutils.rotate_bound(self.contour, degree)


    def get_image_and_contour(self):
        return [self.image, self.contour]


    def show(self):
        cv.imshow(self.path, self.image)
        cv.waitKey(0)
        