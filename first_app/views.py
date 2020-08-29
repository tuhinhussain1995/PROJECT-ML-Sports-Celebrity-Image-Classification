from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages

import numpy as np
import cv2
import os
import pywt
import matplotlib
from matplotlib import pyplot as plt
import joblib
import pickle
import json

import base64
import cv2


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pkldir = os.path.join(BASE_DIR, 'static/ml_files/saved_model.pkl')
jsondir = os.path.join(BASE_DIR, 'static/ml_files/class_dictionary.json')

face = os.path.join(BASE_DIR, 'opencv/haarcascades/haarcascade_frontalface_default.xml')
eye = os.path.join(BASE_DIR, 'opencv/haarcascades/haarcascade_eye.xml')

filedir = ''



def index(request):
    return render(request, 'index.html')


from django.core.files.storage import FileSystemStorage
def uploadPic(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)

        imgdir = os.path.join(BASE_DIR, 'media', name)


        my_text = get_base64_encoded_image(imgdir)

        global filedir
        filedir = os.path.join(BASE_DIR, 'media', 'tuhin.txt')
        with open(filedir, mode='w') as showLine:
            showLine.write("data:image/jpeg;base64,")
            showLine.write(my_text)


        load_saved_artifacts()
        myList = classify_image(get_b64_test_image_for_virat(), None)

        print(myList)

        if len(myList) > 1:
            messages.info(request, 'More Than One Image Has Detected. Better to Upload only One Person Image.')

        if len(myList) < 1:
            messages.info(request, 'Sorry, No Face Has Detected. Please Choose an Image with Two Cleared Eyes.')
            return render(request, 'index.html', {})

        winner = myList[0]['class']

        messi = myList[0]['class_probability'][0]
        maria = myList[0]['class_probability'][1]
        roger = myList[0]['class_probability'][2]
        serena = myList[0]['class_probability'][3]
        virat = myList[0]['class_probability'][4]

        perc = '%'

        return render(request, 'index.html', {'messi' : messi, 'maria' : maria, 'roger' : roger, 'serena' : serena, 'virat' : virat, 'image':imgdir, 'winner' : winner, 'perc':perc})







__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode('utf-8')

    return my_string


def get_b64_test_image_for_virat():
    with open(filedir) as f:
        return f.read()


def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H




result = []

def classify_image(image_base64_data, file_path=None):

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    global result
    result = []

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result



def class_number_to_name(class_num):
    return __class_number_to_name[class_num]



def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(jsondir, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(pkldir, 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")



def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img




cropped_faces = []

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(face)
    eye_cascade = cv2.CascadeClassifier(eye)

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    global cropped_faces
    cropped_faces = []

    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces
