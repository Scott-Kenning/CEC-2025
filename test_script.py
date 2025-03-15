import os
import predict
import csv

def generate_test_case_csv(testing_folder):
    ### Creates/Opens CSV file and writes the filename/tumour present to it
    ### for each image in the CEC_Test folder.
    with open('CEC_output.csv', mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Tumor Present?", "Probability %"])
        for file in os.listdir(testing_folder):
            # Get the prediction if an image contains a tumor or not.
            prediction, probability = predict.predict_image_for_test_suit(os.path.join(testing_folder, file))
            is_tumor = ''
            if(prediction == 0):
                is_tumor = 'no'
            else:
                is_tumor = 'yes'
            writer.writerow([file, is_tumor, round(probability * 100, 2)])

if __name__ == "__main__":
    ### Got these lines of code provided by the CEC 2025 
    ### Programming Environment Variable Setup Guide
    dataset_folder = os.getenv('CEC_2025_dataset')
    if not dataset_folder:
        print("Please set the 'CEC_2025_dataset' environment variable.")
        exit()

    testing_folder = f"{dataset_folder}/CEC_Test/"
    generate_test_case_csv(testing_folder)