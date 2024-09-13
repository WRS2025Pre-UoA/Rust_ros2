from setuptools import setup

package_name = 'rust'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = rust.test_publisher:main',
            'image_publisher = rust.image_publisher:main',  # スクリプトのパスと関数名
            'image_subscriber = rust.image_subscriber:main',
        ],
    },
)