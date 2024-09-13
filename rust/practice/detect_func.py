import cv2

img = cv2.imread("/home/ros/ros2_ws/src/rust/PNG_E/Pattern_1/1.5F_Pole/az30.png")

dim = (1280,720)
img = cv2.resize(img,dim)

cv2.imshow("iimg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("1280-720.png",img)