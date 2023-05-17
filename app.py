import streamlit as st
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO

DEFAULT_STAMP_PATH = "/Users/tatsuya/Python/Streamlit/sticker.png"  # Default stamp image path

# スタンプ画像のアップロード
st.title("顔認識スタンプアプリ")
st.sidebar.title("スタンプ画像のアップロード")
uploaded_stamp = st.sidebar.file_uploader("Choose a stamp image file", type=["jpg", "png", "bmp", "gif"])
if uploaded_stamp is not None:
    stamp_img = Image.open(uploaded_stamp)
else:
    stamp_img = Image.open(DEFAULT_STAMP_PATH)

# 入力画像のアップロード
st.sidebar.title("入力画像のアップロード")
uploaded_image = st.sidebar.file_uploader("Choose an image file", type=["jpg", "png", "bmp", "gif"], key="uploaded_image")
if uploaded_image is not None:
    input_img = Image.open(uploaded_image)
    st.image(input_img, caption="Uploaded Image.", use_column_width=True)

# 顔認識とスタンプ処理
if uploaded_image is not None:
    if st.sidebar.button("Process Image"):
        # face_recognitionに適用できるように画像を変換
        img_array = np.array(input_img)
        stamp_array = np.array(stamp_img)

        # 顔認識
        face_locations = face_recognition.face_locations(img_array, model="cnn")
        
        # スタンプ処理
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top

            # スタンプ画像を顔の大きさにリサイズ（1.5倍する）
            resized_stamp = stamp_img.resize((int(face_width * 1.5), int(face_height * 1.5)))
            
            # スタンプ画像を元の画像に貼り付け
            input_img.paste(resized_stamp, (left, top), resized_stamp)

        # 画像の表示
        st.image(input_img, caption="Processed Image.", use_column_width=True)
        # 画像のダウンロード
        img_temp = BytesIO()
        input_img.save(img_temp, format="JPEG")
        img_temp.seek(0)
        st.download_button('Download Processed Image', img_temp, file_name='processed_image.jpg', mime='image/jpeg')
