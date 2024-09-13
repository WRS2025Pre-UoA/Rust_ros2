import cv2
import numpy as np

def adapt(img):
    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 10)
    # threshold_value = 50
    # _, result = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("adapt",thresh)
    cv2.waitKey(0)

def extract_contours(img,img1):
    # グレースケールに変換
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 画像を二値化する（すでに二値化されている場合はこのステップをスキップ）
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 形態学的な操作を行うためのカーネル（ノイズ除去の強さはカーネルサイズで調整）
    kernel = np.ones((5,5), np.uint8)  # カーネルサイズを調整可能

    # オープニング処理（収縮→膨張でノイズ除去）
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # さらにエッジを綺麗にするために収縮処理を追加
    eroded_image = cv2.erode(cleaned_image, kernel, iterations=1)

    # 輪郭を抽出
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 抽出した輪郭を新しい画像に描画
    output_image = np.zeros_like(image)  # 黒い背景画像を作成
    cv2.drawContours(output_image, contours, -1, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(img1, img1, mask=output_image)

    # 結果を表示
    cv2.imshow('Cleaned Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

def main():
    # 画像を読み込む
    # img = cv2.imread('/home/ros/ros2_ws/W_652_H_652_A_51.png')  # 画像のパスを指定
    # img = cv2.imread('/home/ros/ros2_ws/W_1052_H_1052_A_45.png')  # 画像のパスを指定
    # img1 = cv2.imread('/home/ros/ros2_ws/W_1052_H_1052.png')  # 画像のパスを指定
    img = cv2.imread('/home/ros/ros2_ws/W_648_H_648_A_54.png')  # 画像のパスを指定
    img1 = cv2.imread('/home/ros/ros2_ws/W_648_H_648.png')  # 画像のパスを指定
    
    processed_img = extract_contours(img,img1)
    adapt(processed_img)


if __name__ == '__main__':
    main()
