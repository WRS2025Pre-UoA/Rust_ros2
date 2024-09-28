import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # 画像データを受信
from std_msgs.msg import Float64 #結果を返す
from cv_bridge import CvBridge,CvBridgeError
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
import detect
# キー入力を受け付けて処理を開始したい 好きなタイミングで行う
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image, 'image_topic', self.image_callback, 0)
        self.image_publisher = self.create_publisher(Image, 'rust_result_image', 10)
        self.value_publisher = self.create_publisher(Float64, 'rust_result_value', 10)
        # self.result_publisher = self.create_publisher(String, 'result_topic', 10)
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

            N = detect.create_trackbar(warped)
            area = detect.calc(N)

            print(area)

            text = f"Rust {area} %"
            # フォントの種類 (例: cv2.FONT_HERSHEY_SIMPLEX)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # フォントのスケール（大きさ)
            font_scale = 1

            # テキストの色 (B, G, R)
            color = (0, 0, 255)  # 白色

            # 線の太さ
            thickness = 2
            cv2.putText(warped, text, (50,100), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.imshow("result",warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # 処理後、ノードをシャットダウンするか、トピックからの購読を解除
            # self.get_logger().info('Image processed successfully, shutting down node.')
            if area != None:
                result_value = Float64()
                result_value.data = area
                self.value_publisher.publish(result_value)
                ros_image = self.bridge.cv2_to_imgmsg(warped, 'bgr8')
                self.image_publisher.publish(ros_image)

            

        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
        except Exception as e:
            self.get_logger().error(f'Failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
