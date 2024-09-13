import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # 画像データを送信
from std_msgs.msg import String  # テキスト結果を受け取る
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)
        self.create_subscription(String, 'result_topic', self.result_callback, 10)
        self.bridge = CvBridge()

        # 画像が入っているディレクトリ
        self.image_dir = '/home/ros/ros2_ws/src/rust/PNG_E/az0/'  # 適切なパスに変更
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.image_index = 0

        # 初回の画像送信
        if self.image_files:
            self.send_image()
        else:
            self.get_logger().info('No images found in the directory.')

    def send_image(self):
        if self.image_index < len(self.image_files):
            # 画像ファイルのパスを取得
            image_path = os.path.join(self.image_dir, self.image_files[self.image_index])
            self.get_logger().info(f'Sending image: {self.image_files[self.image_index]}')

            # 画像を読み込み、ROSメッセージに変換
            cv_image = cv2.imread(image_path)
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

            # トピックにパブリッシュ
            self.publisher_.publish(ros_image)
        else:
            self.get_logger().info('All images have been sent.')

    def result_callback(self, msg):
        # サブスクライバーから結果を受け取る
        self.get_logger().info(f'Received result: {msg.data}')
        self.image_index += 1
        if self.image_index < len(self.image_files):
            self.send_image()
        else:
            self.get_logger().info('All images processed, shutting down.')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
