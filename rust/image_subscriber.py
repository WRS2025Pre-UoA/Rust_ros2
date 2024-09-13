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

            #画像の処理
            img = detect.resize_func(cv_image)
            warped = detect.extract_test_piece(img,self.points_list)

            #正方形に切り抜いた画像の縦横サイズ
            h,w = warped.shape[:2]
            N = f"W_{w}_H_{h}.png"
            cv2.imwrite(N,warped)
            #面積比計算
            area,mask = detect.adapt(warped)

            # 結果のサイズを計算し、テキストメッセージとしてパブリッシュ
            result_msg = String()
            result_msg.data = f'Width: {w}, Height: {h}, Area: {area}'
            self.result_publisher.publish(result_msg)
            S = f"W_{w}_H_{h}_A_{int(area)}.png"
            cv2.imwrite(S,mask)

        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
