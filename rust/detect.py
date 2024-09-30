import cv2
from matplotlib import pyplot as plt
import numpy as np

# クリックした点を格納するリスト
# points_list = []

def resize_func(img):
    # width = 1280
    # h,w = img.shape[:2]

    # aspect_ratio= h / w
    # new_height = int(width * aspect_ratio)

    # resized_img = cv2.resize(img, (width,new_height),interpolation=cv2.INTER_AREA)

    resized_img = cv2.resize(img, (640,480),interpolation=cv2.INTER_AREA)

    return resized_img

# 2点間の距離を計算する関数
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# マウスイベント処理関数
def mouseEvents(event, x, y, flags, points_list):
    #左クリックした場合、その座標を保管
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points_list.append([x, y])

def adapt(img):#赤色をマスクし、大津の二値化を用いて領域けんしゅつ

    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # dst_cv = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,1001,20)
    # cv2.imshow("image",dst_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_rust = np.array([0, 30, 0])   # 下限値 51 41 28
    # upper_rust = np.array([102, 255, 100]) # 上限値
    # mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # lower_rust1 = np.array([103, 30, 20])   # 下限値
    # upper_rust1 = np.array([204, 255, 10]) # 上限値
    # mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)

    # lower_rust4 = np.array([205, 30, 10])   # 下限値
    # upper_rust4 = np.array([255, 255, 10]) # 上限値
    # mask4 = cv2.inRange(hsv, lower_rust4, upper_rust4)

    # rust_mask = mask | mask1 | mask4
    # # cv2.imwrite("image.png",dst_cv)

    # black_area = np.sum(rust_mask == 255)
    # white = np.sum(rust_mask == 0)
    # size = black_area + white
    # # print(size)
    # # print(black_area, white )
    # area = black_area / size * 100
    # # print(area)
    # return area,rust_mask

    img = cv2.resize(img,(648,648),interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #サビの赤色、濃い赤色などをマスクする

    lower_rust = np.array([0, 100, 30])   # 下限値 51 41 28
    upper_rust = np.array([102, 255, 125]) # 上限値
    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    lower_rust1 = np.array([103, 100, 30])   # 下限値
    upper_rust1 = np.array([204, 255, 125]) # 上限値
    mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)

    lower_rust4 = np.array([205, 100, 30])   # 下限値
    upper_rust4 = np.array([255, 255, 125]) # 上限値
    mask4 = cv2.inRange(hsv, lower_rust4, upper_rust4)

    rust_mask = mask | mask1 | mask4
    # cv2.imshow("rust",rust_mask)
    # cv2.waitKey(0)
    # オリジナル画像とマスクを使ってサビ部分を抽出
    rust_extract = cv2.bitwise_or(img, img, mask=rust_mask)

    #グレースケール
    gray_rust = cv2.cvtColor(rust_extract, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_rust1",gray_rust)

    # thresh = cv2.adaptiveThreshold(gray_rust, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1001, 1)
    #大津の二値化
    ret2,thresh = cv2.threshold(gray_rust, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # cv2.imshow("gray_rust",thresh)
    # cv2.waitKey(0)

    #サビの白色、その他の黒　領域比
    black_area = np.sum(thresh == 255)
    white = np.sum(thresh == 0)
    size = black_area + white
    # print(size)
    # print(black_area, white )
    area = black_area / size * 100
    return area,thresh


def extract_test_piece(img,points_list):
    # 画像を表示してマウスイベントを設定
    cv2.imshow("Select the 4 points", img)
    cv2.setMouseCallback("Select the 4 points", mouseEvents,points_list)
    # 十分なクリックが行われるまで待機
    while len(points_list) < 4:

        key=cv2.waitKey(1)  # 小さな待機時間で処理を継続する
        if key==-1 and cv2.getWindowProperty("Select the 4 points",cv2.WND_PROP_VISIBLE)<1:
            raise ValueError("Closed Window!")


    cv2.destroyAllWindows()

    # クリックした4つの点を取得
    points = np.array(points_list, dtype="float32")

    # print("Selected points:", points)


    # 最も長い辺の長さを計算
    lengths = [dist(points[i], points[(i+1) % 4]) for i in range(4)]
    max_length = int(max(lengths))

    # 正方形のターゲット座標を設定
    square = np.array([[0, 0], [max_length, 0], 
                    [max_length, max_length], [0, max_length]], dtype="float32")

    # 射影変換行列を計算
    M = cv2.getPerspectiveTransform(points, square)

    # 変換後の画像サイズ
    output_size = (max_length, max_length)

    # 射影変換を適用
    warped = cv2.warpPerspective(img, M, output_size)

    
    return warped

def create_trackbar(warped_img):
    cv2.namedWindow('Warped Image')
    # トラックバーを作成
    cv2.createTrackbar('Brightness', 'Warped Image', 100, 200, lambda x: None)
    cv2.createTrackbar('Contrast', 'Warped Image', 100, 300, lambda x: None)
    # cv2.createTrackbar('Canny Thresh 1', 'Warped Image', 100, 500, lambda x: None)
    # cv2.createTrackbar('Canny Thresh 2', 'Warped Image', 200, 500, lambda x: None)
    # cv2.createTrackbar('Max Line Gap', 'Warped Image', 20, 150, lambda x: None)

    while True:
        update_image(warped_img)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            # save_image()  # 引数を渡さずに呼び出す
            break

    cv2.destroyAllWindows()
    return current_result_image

def update_image(val):
    """
    トラックバーの値に基づいて画像を更新するコールバック関数。
    """
    brightness = cv2.getTrackbarPos('Brightness', 'Warped Image') - 100
    contrast = cv2.getTrackbarPos('Contrast', 'Warped Image') / 100.0
    # canny_threshold1 = cv2.getTrackbarPos('Canny Thresh 1', 'Warped Image')
    # canny_threshold2 = cv2.getTrackbarPos('Canny Thresh 2', 'Warped Image')
    # max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Warped Image')
    
    temp_image = cv2.convertScaleAbs(val, alpha=contrast, beta=brightness)
    # result_image = detect_and_measure_lines(temp_image.copy(), canny_threshold1, canny_threshold2, max_line_gap, scale=20)
    
    # 画像を表示
    cv2.imshow('Warped Image', temp_image)
    global current_result_image
    current_result_image = temp_image

def calc(img):
    h,w = img.shape[:2]
    all = h*w
    white = np.sum(np.all(img == 255,axis=-1))
    area = (all-white) / all * 100
    return area


def main():

    global new_img
    # 画像を読み込み
    # img = cv2.imread("../PNG_E/Pattern_1/1.5F_Pole/az-30.png")  # 画像のパスを指定
    # img = cv2.imread("/home/ros/ros2_ws/src/rust/PNG_E/Pattern_3/1.5F_Wall/az0.png")
    img = cv2.imread("/home/ros/ros2_ws/src/rust/PNG_E/az0/P4_az0.png")
    # img = cv2.imread("../PNG_E/Pattern_1/2F_Wall/az-30.png")  # 画像のパスを指定
    img = resize_func(img)
    points_list = []

    new_img=extract_test_piece(img,points_list)

    N=create_trackbar(new_img)
    cv2.imshow("Warped1",N)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    num = calc(N)
    print(num)
 


if __name__ == '__main__':
    main()
