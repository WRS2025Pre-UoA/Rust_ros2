import cv2
from matplotlib import pyplot as plt
import numpy

def dist(p1,p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process():
    # img = cv2.imread("../PNG_E/Pattern_1/1.5F_Pole/az30.png",cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("../PNG_E/Pattern_1/1.5F_Pole/az0.png")
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    dim = (1280,720)
    # img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1,dim,interpolation=cv2.INTER_AREA)

    # cv2.imshow("image",img)
    # cv2.imshow("image1",img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    equalized = cv2.equalizeHist(img1)

    dst_cv1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 301, 31)
    dst_cv2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 291, 30)

    dst_cv3 = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 301, 31)
    dst_cv4 = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 291, 30)

    cv2.imshow("image1",dst_cv1)
    cv2.imshow("image2",dst_cv2)
    cv2.imshow("image3",dst_cv3)
    cv2.imshow("image4",dst_cv4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

    



def main():
    process()

if __name__ == '__main__':
    main()
 