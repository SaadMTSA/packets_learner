from setuptools import setup, find_packages

setup(
    name='packets_learner',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        packets_learner=src.detector:cli
        """,
)