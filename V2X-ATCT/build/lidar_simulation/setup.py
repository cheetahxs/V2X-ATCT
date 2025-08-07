from setuptools import setup,find_packages

if __name__ == '__main__':
      setup(name='lidar_simulation',
            install_requires=['numpy',
                              "open3d",
                              'math',
                              "astropy",
                              "yaml",
                              ],
            python_requires=">=3.7",
            version='1.0',
            description='V2XGen: Generate Realistic Test Scenes for V2X Communication Systems.',
            author='JackyLljk',
            include_package_data=True,
            packages=find_packages()
      )