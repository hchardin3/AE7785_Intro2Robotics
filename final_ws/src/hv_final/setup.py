from setuptools import find_packages, setup

package_name = 'hv_final'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/final.launch.py']),  # Include launch files
        ('share/' + package_name + '/data', ['hv_final/svm_class_2024_1.joblib']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='burger',
    maintainer_email='burger@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_control = hv_final.robot_control:main',
            'sign_vision = hv_final.sign_vision:main',
        ],
    },
)
