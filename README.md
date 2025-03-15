# Requirements
1. Python 3.11 (any subversion should work)
2. The libraries in ```requirements.txt```

# How to Run
After installing Python, follow the steps to install the requirements and run the test file.

1. Ensure you have a system environment variable named ```CEC_2025_dataset``` that points to the ```CEC_2025/``` folder containing the ```CEC_test/``` folder.

2. From the top-level of the repository, run the following code to install the requirements:
```pip install -r requirements.txt```

3. Once the ```CEC_test/``` folder is populated with test images, run the test_script.py file: ```python test_script.py```.
4. A CSV file named ```CEC_output.csv``` should now exist in the top-level repository with output results for each image.
