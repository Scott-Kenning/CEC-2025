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

## Running the website
- In the root folder, run `python -m uvicorn main:app --reload`
- create a file called .env in `/website` with the following line: `API_URL = "http://127.0.0.1:8000/classify"`
- In a separate terminal in the website folder, run `npm install && npm run dev`
