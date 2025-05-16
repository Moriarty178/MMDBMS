import os
import numpy as np
import psycopg2
import cv2
import sys
import time
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.feature.texture import graycomatrix, graycoprops
from colorama import Fore, Back, Style, init
from datetime import datetime
from tqdm import tqdm




# Kết nối PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname="recognize_office_equipment", user="postgres", password="17082002", host="localhost"
    )

# Chuẩn hóa kích thước ảnh
IMG_SIZE = (256, 256)

EXPECTED_FEATURE_SIZE = 1044

# Trích xuất Histogram HSV
def extract_histogram(image):
    image = cv2.resize(image, IMG_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    # return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()

    # Đảm bảo histogram luôn có 512 giá trị bằng cách padding nếu cần
    return np.pad(hist, (0, max(0, 512 - len(hist))), mode='constant')

# Trích xuất GLCM (Grey Level Co-occurrence Matrix)
def extract_glcm(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    levels = 64
    # gray = (gray / 256 * levels).astype(np.uint8)
    gray = np.clip((gray / 256 * levels), 0, levels - 1).astype(np.uint8)


    #  Tính GLCM theo nhiều góc angels=[0, pi/4, pi/2, 3pi/4]
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3 * np.pi/4], levels=levels, symmetric=True, normed=True)

    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)


    features = np.array(features)
    assert features.shape[0] == 20, f"GLCM feature vector size mismatch: {features.shap}"
    return features

# Trích xuất HOG (Histogram of Oriented Gradients)
def extract_hog(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    # return features[:256] # Giới hạn vector
    # features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    # return np.pad(features[:256], (0, max(0, 256 - len(features))), mode='constant')

    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    features = np.pad(features, (0, max(0, 256 - len(features))), mode='constant')[:256]  
    assert features.shape[0] == 256, f"HOG feature vector size mismatch: {features.shape}"
    return features
    

# Trích xuất ORB (Oriented FAST and Rotated BRIEF)
def extract_orb(image):
    image = cv2.resize(image, IMG_SIZE)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # if descriptors is not None:
    #     flat_desc = descriptors.flatten()
    #     return np.pad(flat_desc, (0, max(0, 256 - len(flat_desc))), mode='constant')
    #     # return descriptors.flatten()[:256]  # Giới hạn kích thước vector. TH này có thể dẫn đến kích thước không đồng nhất do descriptors < 256 => dùng cách trên: Thêm padding 0 nếu len < 256
    # return np.zeros(256)  # Nếu không có keypoints
    if descriptors is not None:
        flat_desc = descriptors.flatten()
        flat_desc = np.pad(flat_desc, (0, max(0, 256 - len(flat_desc))), mode='constant')[:256]
    else:
        flat_desc = np.zeros(256)
        
    assert flat_desc.shape[0] == 256, f"ORB feature vector size mismatch: {flat_desc.shape}"
    return flat_desc


# Lưu vector đặc trưng vào PostgreSQL
def save_feature(image_path, label, feature_vector):
    conn = connect_db()
    cursor = conn.cursor()
    feature_vector_str = "{" + ",".join(map(str, feature_vector)) + "}"
    cursor.execute("""
        INSERT INTO image_features (image_path, label, feature_vector) 
        VALUES (%s, %s, %s)
    """, (image_path, label, feature_vector_str))
    conn.commit()
    conn.close()

# Duyệt qua dataset và trích xuất đặc trưng
def process_dataset(dataset_path):
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if os.path.isdir(category_path):
                for img_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
                    img_path = os.path.join(category_path, img_name)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    hist_features = extract_histogram(image)
                    glcm_features = extract_glcm(image)
                    hog_features = extract_hog(image)
                    orb_features = extract_orb(image)
                    
                    # feature_vector = np.concatenate((hist_features, glcm_features, hog_features, orb_features))
                    feature_vector = np.concatenate((hist_features, glcm_features, hog_features, orb_features))
                    assert feature_vector.shape[0] == EXPECTED_FEATURE_SIZE, f"Feature vector size mismatch: {feature_vector.shape}"

                    save_feature(img_path, category, feature_vector)

# Chạy trích xuất đặc trưng
# Thời gian bắt đầu chạy
# start_time = time.perf_counter()
# process_dataset("D:/CSDLDPT/recognize_office_equipment/dataset")
# end_time = time.perf_counter()
# total_time = end_time - start_time
# print(f"Hoàn thành trích xuất đặc trưng và lưu vào PostgreSQL!. Thời gian bắt đầu = {start_time}. Thời gian kết thúc = {end_time}. Tổng thời gian = {total_time}s.")



# Khởi tạo colorama


# Hiển thị theo kiểu trò chơi điện tử
def print_game_style(start_time, end_time, total_time):
    print(f"""
        {Fore.YELLOW + Back.BLACK + Style.BRIGHT}==========================================================
        {Fore.GREEN}||🚀🚀🚀 HOÀN THÀNH TRÍCH XUẤT ĐẶC TRƯNG... ✨          ||
        {Fore.CYAN}|| Thời gian bắt đầu: {start_time}               ||
        {Fore.MAGENTA}|| Thời gian kết thúc: {end_time}              ||
        {Fore.RED}|| Tổng thời gian: {total_time}s                      |||||
        {Fore.YELLOW + Back.BLACK + Style.BRIGHT}==========================================================
    """)

if __name__ == "__main__": # mã nằm ngoài "__main__" sẽ được chạy khi nó được các file khác import (vd: file A import file B để chạy 1 số func, nhưng nó sẽ chạy cả những func của B mà nằm ngoài "__main__")
    init(autoreset=True)

    start_time = datetime.now()
    print(Fore.CYAN + "✨ Hệ thống trích xuất đặc trưng đang hoạt động... ✨")
    process_dataset("D:/CSDLDPT/recognize_office_equipment/dataset")
    end_time = datetime.now()
    total_time = end_time - start_time

    # Gọi hàm hiển thị theo kiểu trò chơi
    print_game_style(start_time.strftime('%d-%m-%Y %H:%M:%S'), end_time.strftime('%d-%m-%Y %H:%M:%S'), total_time)

    # Hiển thị kết quả thêm vào
    print(Fore.GREEN + "Hoàn thành trích xuất đặc trưng và lưu vào PostgreSQL!")