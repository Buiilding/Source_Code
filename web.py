# -*- coding: utf-8 -*-
import os
import streamlit as st
from PIL import Image
import torch #operated by facebook with a lot of library
import torch.nn as nn
import Inference_square
weight_path = 'C:/Users/tuana/SLC_PRO/Source_Code/best.pth'
from Classification_Model import Model

model = Model(27)
import pandas as pd
table = pd.read_csv('C:/Users/tuana/SLC_PRO/Source_Code/data_predict.csv')
base_dir = "C:/Users/tuana/SLC_PRO/Source_Code/Data_Sample"
def data():
    st.title("Dataset" )
    st.write("Đây là dataset được dùng để huấn luyện model")
    st.write("Đây là sản phẩm trên trang web Kaggle.com")
    # Load and display the images
    # Define the base directory where the class folders arste located
    names = ['0','1','2','3','4','5','6','7','8','9','a','b','bye','c','d','e','good','good morning','hello','little bit','no','pardon','please','project','whats up','yes']

    for x, image_folder in enumerate(names):
        class_name = os.path.join(base_dir, image_folder)
        image_files = os.listdir(class_name)
        
        if len(image_files) > 0:
            image_file = os.path.join(class_name, image_files[0])
            image = Image.open(image_file)
            st.image(image, caption=f"Class {x}", use_column_width=True)
        else:
            st.write(f"No image found for Class {x}")

# Define a function for the accuracy page
def accuracy():
    st.title("Accuracy")
    st.write("Đây là bảng hiển thị độ chính xác của model:")
    st.write("Độ chính xác tương đối: 94 phần trăm")
    st.dataframe(table)

# Define a function for the dashboard page
def dashboard():
    st.title("Sign Language Classification Model")
    st.write("Chào mừng tới với AI Computer Vision Model website của tôi!")
    st.write("Đây là một số thông tin về AI model của tôi:")
    st.write("Model name: SLC")
    st.write("Model description: Đây là phần mềm dịch các ký tự thủ ngữ sang chữ cái hoặc chữ số. Website sẽ yêu cầu người dùng cung cấp hình ảnh có chứa cử chỉ thủ ngữ rồi AI model sẽ phân tích hình ảnh và đưa ra kết quả.")
    st.write("Mục đích phát triển: Mục đích chính của việc phát triển CIGNS - Sign Language AI Model là tạo ra một hệ thống công nghệ nhận diện ngôn ngữ ký hiệu (sign language) tự động và chính xác. Mô hình này nhằm giúp người dùng dễ dàng giao tiếp với nhau và với cộng đồng người điếc thông qua việc chuyển đổi ngôn ngữ ký hiệu thành văn bản hoặc âm thanh.")
    st.write("<span style='color: red;'>HƯỚNG DẪN: Hãy bấm vào phần dataset để chọn hình ảnh mà bạn muốn tự chụp rồi hướng đến phần AI Computer Vision để insert ảnh của bạn vào</span>", unsafe_allow_html=True)
    
    
# Define a function for the AI computer vision page
def ai_computer_vision(model):
    st.title("Sign Language Classification Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Pass the uploaded image to your inference function
        top_k_classes = Inference_square.inference(model, weight_path, uploaded_file, 27)
        names = ['0','1','2','3','4','5','6','7','8','9','NULL','a','b','bye','c','d','e','good','good morning','hello','little bit','no','pardon','please','project','whats up','yes']
        # Get the predicted class name
        predicted_class = names[top_k_classes[0]]
        # Format the output with bigger and red text
        result_text = f"Cử chỉ ký hiệu được cho trong hình được định nghĩa là: <span style='font-size: 24px; color: red;'>{predicted_class}</span>"
        st.markdown(result_text, unsafe_allow_html=True)




# Create a menu for navigation between pages
st.sidebar.title("Menu")
page = st.sidebar.radio("Navigation", ["Dashboard","Accuracy","Dataset","AI Computer Vision"])

# Display the selected page
if page == "Dashboard":
    dashboard()
elif page == "AI Computer Vision":
    ai_computer_vision(model)
elif page == "Accuracy":
    accuracy()    
elif page == "Dataset":
    data()