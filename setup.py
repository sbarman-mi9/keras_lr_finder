from setuptools import setup

setup(name='keras_lr_finder',
      version='0.1',
      description='Learning rate finder for Keras.',
      url='https://github.com/sbarman-mi9/keras_lr_finder',
      author='Snehasish Barman',
      author_email='sbarman@wpi.edu',
      license='MIT',
      packages=['keras_lr_finder'],
      install_requires=[
            'tensorflow-gpu>=2.0.0rc0',
            'matplotlib'
      ],
      zip_safe=False)