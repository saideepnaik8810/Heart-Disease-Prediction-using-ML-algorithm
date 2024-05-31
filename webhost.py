from flask import Flask,render_template,request
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from base64 import b64encode
import nibabel as nib
import matplotlib.pyplot as plt
import os
import io
import random
from skimage.transform import resize
from skimage.util import img_as_float,img_as_ubyte
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

scaler=MinMaxScaler()

app=Flask(__name__)

model=pickle.load(open('Knnmodel_sec.pkl','rb'))
cnnmodel=load_model('Arrhythmia.h5')
segmodel=load_model('ACDC_3d_take4.hdf5',compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/14form')
def form():
    return render_template('14form.html')

@app.route('/predict_if',methods=["POST"])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('14form.html',output=output)

@app.route('/13parainfo')
def info():
    return render_template('13parainfo.html')

@app.route('/ArrhyPred')
def predictArrhythmia():
    return render_template('ArrPred.html')

@app.route('/DetectArr',methods=['POST'])
def ArrDetect():
    image=request.files['file']
    img_np = np.fromstring(image.read(), np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    testimg=cv2.resize(image,(224,224))
    testimg=testimg.reshape((1,224,224,3))

    pred=cnnmodel.predict(testimg)
    y_pred = np.argmax(pred, axis=1)
    output=y_pred[0]

    #code to display image in frontend
    img_encoded = cv2.imencode('.png', image)[1].tobytes()
    img_base64 = b64encode(img_encoded).decode('utf-8')
    return render_template('ArrPred.html',output=output,img_base64=img_base64)

@app.route('/Segmentation')
def formseg():
    return render_template('Segmentation.html')

#function for clahe normalisation
def apply_clahe(image_slice):
    # Convert the image slice to float and rescale it to the range [0,1]
    image_slice = img_as_float(image_slice)
    image_slice = exposure.rescale_intensity(image_slice, out_range=(0, 1))
    
    # Apply CLAHE normalization with specified clip limit and tile grid size
    clahe_normalized_slice = exposure.equalize_adapthist(image_slice, clip_limit=0.03, nbins=256)
    
    # Convert the CLAHE-normalized slice to uint8 format
    return img_as_ubyte(clahe_normalized_slice)
    

@app.route('/segimage',methods=['POST'])
def segimage():
    upload_path="temp.nii"
    
    segimg=request.files['numfile']
    
    segimg.save(upload_path)

    img=nib.load(upload_path)
    img_data=img.get_fdata()
    os.remove(upload_path)

####################################################################################
    #image_rescaling
    current_spacing = img.header.get_zooms()

    # Calculate the scaling factors for resampling
    scaling_factors = [current_spacing[0] / 1.0, current_spacing[1] / 1.0, current_spacing[2]]

    # Resample the image data using interpolation
    resampled_data = resize(img_data, (int(img_data.shape[0] * scaling_factors[0]),
                                       int(img_data.shape[1] * scaling_factors[1]),
                                       int(img_data.shape[2] * scaling_factors[2])),
                            preserve_range=True)

    # Update the pixel spacing information
    new_spacing = (1.0, 1.0, current_spacing[2])  # Keep the original slice thickness
    img.header.set_zooms(new_spacing)
###################################################################################################


    #CLAHE normalisation

    
    clahe_normalized_data = np.zeros_like(resampled_data)
    # Iterate through each slice in the z-axis
    for i in range(resampled_data.shape[2]):
        # Apply CLAHE to the current slice
        clahe_normalized_data[:, :, i] = apply_clahe(resampled_data[:, :, i])

    
    


##############################################################################################
    
    #Crop the image
    resampled_shape = clahe_normalized_data.shape

    # Calculate the starting and ending indices for cropping in each dimension
    start_x = (resampled_shape[0] - 224) // 2
    end_x = start_x + 224
    start_y = (resampled_shape[1] - 224) // 2
    end_y = start_y + 224

    # Perform center cropping
    temp_frame = clahe_normalized_data[start_x:end_x, start_y:end_y, :16]
    #normalise to 0-1
    temp_frame=scaler.fit_transform(temp_frame.reshape(-1, temp_frame.shape[-1])).reshape(temp_frame.shape)
    temp_frame=np.stack([temp_frame]*3,axis=3)

    #preparing image for prediction
    test_img_input = np.expand_dims(temp_frame, axis=0)
    test_prediction = segmodel.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


    plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace=0.3)

    n=random.randint(0,temp_frame.shape[2]-1)
    plt.subplot(231)
    plt.title('MRI image')
    plt.imshow(temp_frame[:,:,n,1],cmap='gray')
    plt.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plot_base64_1 = b64encode(buffer.read()).decode('utf-8')

    #free up buffer
    buffer.seek(0)
    buffer.truncate()

    plt.clf()

    #plot the second graph
    plt.subplot(232)
    plt.title('Prediction')
    plt.imshow(test_prediction_argmax[:,:,n])
    plt.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plot_base64_2 = b64encode(buffer.read()).decode('utf-8')

    # Close the plot to free up memory
    plt.close()

    return render_template('Segmentation.html',plot_base64_1=plot_base64_1,plot_base64_2=plot_base64_2)

if __name__=='__main__':
    app.run(debug=True)