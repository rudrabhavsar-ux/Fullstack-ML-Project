from setuptools import setup,find_packages
from typing import List

hyphen='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    Reads the Requirements
    '''
    requirement=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if hyphen in requirements:
            requirements.remove(hyphen)
    return requirements

setup(
    name="MLProject",
    version="1.0",
    author="Rudra Bhavsar",
    author_email="rudrabhavsar33@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)