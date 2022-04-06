from setuptools import setup, find_packages

setup(name='BboxToolkit',
      version='1.1',
      description='a tiny toolkit for special bounding boxes',
      author='jbwang1997',
      packages=find_packages(),
      install_requires=[
          'matplotlib>=3.4',
          'opencv-python',
          'terminaltables',
          'pillow',
          'shapely',
          'numpy'])
