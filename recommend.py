
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm




feature_list = np.array(pickle.load(open('featureslist.pkl','rb')))
filenames = pickle.load(open('filenames2.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(244,244,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Product Recommender System')

activiteis = ["Home", "About"]
choice = st.sidebar.selectbox("Select Activity", activiteis)
st.sidebar.markdown(
        """ Developed by Sampanna  
            Email : mishrasampanna1998@gmail.com  
            [LinkedIn] (http://www.linkedin.com/in/sampanna-mishra-0996ab18b/)
            """)
if choice == "Home":
        html_temp_home1 = """<div style="background-color:#0a2342;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Fashion Product Recommendation Application using Transfer Learning model, ML algorithm and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#0a2342;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Fashion Product Recommendation Application using Transfer Learning model, ML algorithm and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#7a9fcc;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Sampanna Mishra using Streamlit Framework, Transfer Learning Model, Machine Learning algorithm for demonstration purpose.If you have any suggestion or wnat to comment just write a mail at mishrasampanna1998@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

else:
   pass


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        # st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

       
        with col1:
            image_1 = Image.open(filenames[indices[0][0]])
            st.image(image_1)
        with col2:
            image_2 = Image.open(filenames[indices[0][1]])
            st.image(image_2)
        with col3:
            image_3 = Image.open(filenames[indices[0][2]])
            st.image(image_3)
        with col4:
            image_4 = Image.open(filenames[indices[0][3]])
            st.image(image_4)
        with col5:
            image_5 = Image.open(filenames[indices[0][4]])
            st.image(image_5)
    else:
        st.header("Some error occured in file upload")