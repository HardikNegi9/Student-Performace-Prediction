from setuptools import setup, find_packages


HYPEN_E_DOT='-e .'

def get_requirements(path):
    with open(path) as f:
        requirements =  f.read().splitlines()
        if(HYPEN_E_DOT in requirements):
            requirements.remove(HYPEN_E_DOT)
        return requirements


setup(
    name='ml_project',
    version='0.0.1',
    author='baezyii',
    author_email='hardiknegi764isc@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)

