from setuptools import find_packages, setup

setup(
    name='student_performance_prediction',
    version='0.0.1',
    author='Senior Data Scientist',
    author_email='example@domain.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'joblib'
    ]
)
