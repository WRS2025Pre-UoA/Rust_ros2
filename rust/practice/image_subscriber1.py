import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # 画像データを受信
from std_msgs.msg import String  # 結果を返す
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
import detect

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image, 'image_topic', self.image_callback, 10)
        self.result_publisher = self.create_publisher(String, 'result_topic', 10)
        self.bridge = CvBridge()

        self.points_list = []

    def image_callback(self, msg):
        try:

            # 画像リセット
            self.points_list = []
            # ROS 画像メッセージを OpenCV 画像に変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # # 画像を縮小するサイズを指定（例えば、幅を640ピクセルにする）
            # scale = 0.3  # 縮小のスケール（0.5で半分のサイズにする）
            # width = int(cv_image.shape[1] * scale)
            # height = int(cv_image.shape[0] * scale)
            # dim = (width, height)

            dim = (1280,720)

            # 画像を縮小
            cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)

            # 画像を表示してマウスイベントを設定
            cv2.imshow("Select the 4 points", cv_image)
            cv2.setMouseCallback("Select the 4 points", self.mouse_events)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # クリックした4つの点を取得
            points = np.array(self.points_list, dtype="float32")
            print("Selected points:", points)

            if len(points) != 4:
                self.get_logger().warn('Four points were not selected. Processing aborted.')
                return

            # 最も長い辺の長さを計算
            lengths = [self.dist(points[i], points[(i+1) % 4]) for i in range(4)]
            max_length = int(max(lengths))

            # 正方形のターゲット座標を設定
            square = np.array([[0, 0], [max_length, 0], 
                               [max_length, max_length], [0, max_length]], dtype="float32")

            # 射影変換行列を計算
            M = cv2.getPerspectiveTransform(points, square)

            # 変換後の画像サイズ
            output_size = (max_length, max_length)

            # 射影変換を適用
            warped = cv2.warpPerspective(cv_image, M, output_size)

            # 結果を表示
            cv2.imshow("Warped Image", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 結果をファイルに保存
            cv2.imwrite("Detect_png.png", warped)

            # 結果のサイズを計算し、テキストメッセージとしてパブリッシュ
            result_msg = String()
            result_msg.data = f'Width: {output_size[0]}, Height: {output_size[1]}'
            self.result_publisher.publish(result_msg)

        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.points_list.append([x, y])

    @staticmethod
    def dist(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
