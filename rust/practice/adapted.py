import cv2
import numpy as np

def extract_rust(img):
    # 画像をHSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # サビ（赤色）に相当する色範囲を設定 (HSV空間)
    lower_rust = np.array([0, 100, 30])   # 下限値 51 41 28
    upper_rust = np.array([102, 255, 125]) # 上限値
    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    lower_rust1 = np.array([103, 100, 30])   # 下限値
    upper_rust1 = np.array([204, 255, 125]) # 上限値
    mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)

    # lower_rust2 = np.array([103, 0, 50])   # 下限値
    # upper_rust2 = np.array([153, 255, 140]) # 上限値
    # mask2 = cv2.inRange(hsv, lower_rust2, upper_rust2)

    # lower_rust3 = np.array([154, 0, 50])   # 下限値
    # upper_rust3 = np.array([204, 255, 140]) # 上限値
    # mask3 = cv2.inRange(hsv, lower_rust3, upper_rust3)

    lower_rust4 = np.array([205, 100, 30])   # 下限値
    upper_rust4 = np.array([255, 255, 125]) # 上限値
    mask4 = cv2.inRange(hsv, lower_rust4, upper_rust4)
    # サビ（赤色）に相当する色範囲を設定 (BGR空間)
    # lower_rust = np.array([50, 0, 0])   # 下限値（BGR）
    # upper_rust = np.array([255, 0, 0]) # 上限値（BGR）
    # mask1 = cv2.inRange(img, lower_rust, upper_rust)
    # cv2.imshow("mask",mask)
    # cv2.imshow("mask1",mask1)
    # cv2.imshow("mask2",mask2)
    # cv2.imshow("mask3",mask3)
    # cv2.imshow("mask4",mask4)
    

    # 赤色のもう一つの範囲（Hの値が180付近もあるため）
    # lower_rust2 = np.array([170, 0, 0])
    # upper_rust2 = np.array([250, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_rust2, upper_rust2)
    # cv2.imshow("mask2",mask2)
    # 両方のマスクを合成
    # rust_mask = mask | mask1 | mask2 | mask3 | mask4
    rust_mask = mask | mask1 | mask4
    cv2.imshow("rust",rust_mask)
    cv2.waitKey(0)
    # オリジナル画像とマスクを使ってサビ部分を抽出
    rust_extract = cv2.bitwise_or(img, img, mask=rust_mask)

    # return rust_extract
    # # cv2.imshow("rust_extract",rust_extract)
    # # グレースケールに変換
    gray_rust = cv2.cvtColor(rust_extract, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_rust1",gray_rust)
    # cv2.waitKey(0)

    # gray_rust = cv2.equalizeHist(gray_rust)

    # # アダプティブスレッショルド
    # thresh = cv2.adaptiveThreshold(gray_rust, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 1001,1)
    ret2,thresh = cv2.threshold(gray_rust, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("gray_rust",thresh)
    cv2.waitKey(0)
    return thresh

def main():
    # 画像を読み込む
    # filename='/home/ros/ros2_ws/W_605_H_605.png'
    # filename='/home/ros/ros2_ws/W_648_H_648.png'
    # filename = "/home/ros/ros2_ws/W_1052_H_1052.png"
    # filename = "/home/ros/ros2_ws/W_934_H_934.png"
    filename="/home/ros/ros2_ws/W_886_H_886.png"
    # filename= "/home/ros/ros2_ws/W_931_H_931.png"だめ
    # filename = "/home/ros/ros2_ws/W_930_H_930.png"だめ
    # img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)  # 画像のパスを指定
    img = cv2.imread(filename)
    img = cv2.resize(img,(648,648),interpolation=cv2.INTER_AREA)
    cv2.imshow("x",img)
    cv2.waitKey(0)
    processed_img = extract_rust(img)

    # 結果を表示
    # cv2.imshow('Extracted Rust', processed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    black_area = np.sum(processed_img == 255)
    white = np.sum(processed_img == 0)
    size = black_area + white
    print(size)
    # print(black_area, white )
    area = black_area / size * 100
    print(area)
    cv2.imwrite("P1_2FW.png",processed_img)


if __name__ == '__main__':
    main()



# img = cv2.imread("Detect_png.png",cv2.IMREAD_GRAYSCALE)
# equalized = cv2.equalizeHist(img)

# dst_cv = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, 40)
                                                    
# cv2.imshow("image1",img)
# cv2.imshow("image",dst_cv)
# cv2.imshow("eq",equalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("image.png",dst_cv)
# cv2.imwrite("balck.png",equalized)

# black_area = np.sum(dst_cv == 0)
# white = np.sum(dst_cv == 255)
# size = black_area + white
# print(size)
# print(black_area, white )
# area = black_area / size * 100
# print(area)