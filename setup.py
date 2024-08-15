from setuptools import setup
import glob
import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", 'r') as f:
    long_description = f.read()

scripts = []
for file in glob.glob('tools4vasp/*'):
    if file.endswith('.py'):
        scripts.append(file)
    elif file.endswith('.sh'):
        scripts.append(file)

setup(
   name='tools4vasp',
   version='1.0',
   description='Python tools for VASP',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   author='Patrick Melix',
   author_email='patrick.melix@uni-leipzig.de',
   url="https://github.com/Tonner-Zech-Group/VASP-tools",
   packages=['tools4vasp'],
   install_requires=['ase', 'numpy', 'matplotlib', 'natsort', 'pymatgen'],
   console_scripts = scripts
)
