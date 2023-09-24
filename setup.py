from setuptools import setup,find_packages
from typing import List

hyphen_e_dot = "-e."

def get_packages(filepath:str)->List[str]:
    requirement = []
    with open(filepath) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace("\n","") for req in requirement]
        if hyphen_e_dot in requirement:
            requirement.remove(hyphen_e_dot)

    return requirement


setup(
    name='Predictive Maintenance Project',
    version='0.0.1',
    description='small general software for Feynn Labs',
    author='Yash Mayur',
    author_email='ysmayur1992@gmail.com',
    packages=find_packages(),
    install_requires=get_packages('requirements.txt')
)