from ast import operator
import streamlit as st
import cv2
import numpy as np
st.title("My first app!")
st.markdown("""
    # Streamlit
    ## 1. Giới thiệu
    ## 2. Thành phần cơ bản           
""")
#2. Text box + 6. Button
# 3. checkbox
# 4. radio button;
# a = st.text_input("Input a")
# b = st.text_input("Input b")
# operator = st.selectbox("Chọn phép tính",
#                        ['Cộng', 'Trừ', 'Nhân', 'Chia'])
# clicked = st.button("Thực hiện")
# if clicked:
#     if operator == "Cộng":
#         st.text_input("Kết quả", int(a) + int(b))
#     elif operator == "Trừ":
#         st.text_input("Kết quả", int(a) - int(b))
#     elif operator == "Nhân":
#         st.text_input("Kết quả", int(a) * int(b))
#     elif operator == "Chia":
#         st.text_input("Kết quả", float(a) / float(b))
# Clicked = st.button("Cộng")
# if Clicked:
#     st.text_input("Kết quả", int(a) + int(b))
    
# 7. Group / tab control
tab1, tab2, tab3, tab4 = st.tabs(["Cat", "Dog", "Owl", "Calculator"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
with tab4:
    a = st.text_input("Input a")
    b = st.text_input("Input b")
    operator = st.selectbox("Chọn phép tính",
                        ['Cộng', 'Trừ', 'Nhân', 'Chia'])
    clicked = st.button("Thực hiện")
    if clicked:
        if operator == "Cộng":
            st.text_input("Kết quả", int(a) + int(b))
        elif operator == "Trừ":
            st.text_input("Kết quả", int(a) - int(b))
        elif operator == "Nhân":
            st.text_input("Kết quả", int(a) * int(b))
        elif operator == "Chia":
            st.text_input("Kết quả", float(a) / float(b))
    Clicked = st.button("Cộng")
    if Clicked:
        st.text_input("Kết quả", int(a) + int(b))
        
# 9. Upload file
upload_file = st.file_uploader("Chọn file")
if upload_file is not None:
    bytes_data = upload_file.getvalue()
    img_path = "data/" +upload_file.name
    with open(img_path, "wb") as f:
        f.write(bytes_data)
    # process image
    I = cv2.imread(img_path, 0)
    
st.snow()
st.balloons()