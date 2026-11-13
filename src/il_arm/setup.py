from glob import glob
from setuptools import setup

package_name = 'il_arm'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deiva',
    description='Imitation learning for pick-and-place: sim expert demos, BC with action chunking, real-arm recorder',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo_recorder = il_arm.demo_recorder_node:main',
            'policy_runner = il_arm.policy_runner_node:main',
            'fake_arm = il_arm.fake_arm:main',
            'collect_demos = il_arm.collect_demos:main',
            'train_bc = il_arm.train_bc:main',
            'eval_policy = il_arm.eval_policy:main',
        ],
    },
)
