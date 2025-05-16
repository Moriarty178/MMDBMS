import numpy as np
import glob
import cv2  # Để đọc ảnh
import heapq  # Dùng heap để tìm 5 ảnh tốt nhất
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog

from tqdm import tqdm
from extract_features_postgresql import connect_db, extract_histogram, extract_glcm, extract_hog, extract_orb

def load_dataset():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT feature_vector, label, image_path FROM image_features")
    dataset = [(np.array(row[0]), row[1], row[2]) for row in cursor.fetchall()]  # Thêm image_path
    conn.close()
    return dataset

def load_weight():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT weights FROM optimal_weights WHERE id = 46")
    weights = cursor.fetchone()  # Lấy dòng mới nhất
    conn.close()
    return np.array(weights[0]) if weights else None  # Trả về numpy array hoặc None

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()

    img_new_path = filedialog.askopenfilename(
        title="Upload your image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")], 
    )

    if not img_new_path:
        print("Không có ảnh nào được chọn => Exit.")
        exit()


    # I. Đọc ảnh đầu vào
    # img_new_path = glob.glob("D:/CSDLDPT/Test/*.jpg")[0]
    img_new = cv2.imread(img_new_path)#, cv2.IMREAD_GRAYSCALE)  # Chuyển ảnh về grayscale

    # Kiểm tra ảnh có tồn tại không
    if img_new is None:
        raise ValueError("Không thể đọc ảnh đầu vào!")

    # II. Trích xuất đặc trưng của ảnh mới
    hist = extract_histogram(img_new)
    glcm = extract_glcm(img_new)
    hog = extract_hog(img_new)
    orb = extract_orb(img_new)

    # Gộp thành vector đặc trưng
    feature = np.concatenate([hist, glcm, hog, orb])

    print(f"Size feature {len(feature)}")
    # III. Tính toán độ tương đồng với ảnh trong database

    # 1. Lấy toàn bộ dataset
    dataset = load_dataset()

    # 2. Lấy bộ trọng số
    w = load_weight()
    if w is None:
        raise ValueError("Không tìm thấy trọng số tối ưu trong database!")
    else:
        print("📊 Thống kê trọng số:")
        print(f"Mean: {np.mean(w)}, Min: {np.min(w)}, Max: {np.max(w)}, Std: {np.std(w)}")
        plt.hist(w, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.xlabel('Weight values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Weight Distribution')
        plt.show()
    
    # 3. Tính toán độ tương đồng và lưu 5 ảnh gần nhất
    best_matches = []  # Heap lưu trữ 5 ảnh tốt nhất
    print(f"Size of dataset: {len(dataset)}")
    for img_vector, label, img_db_path in dataset:
        # Tính toán độ tương đồng
        # similarity = -np.sum(w * np.abs(feature - img_vector))  # Dùng dấu "-" để heap tìm giá trị lớn nhất
        similarity = -np.sum(w * (np.abs(feature - img_vector) / (np.abs(feature) + np.abs(img_vector) + 1e-8)))

        # Lưu vào heap
        heapq.heappush(best_matches, (similarity, img_db_path))

        # Giữ lại 5 ảnh tốt nhất
        if len(best_matches) > 5:
            heapq.heappop(best_matches)

    # 4. Sắp xếp kết quả (từ cao đến thấp)
    best_matches.sort(reverse=True, key=lambda x: x[0])

    # In ra 5 ảnh tương đồng nhất
    print("📌 Top 5 ảnh tương đồng nhất:")
    for score, img_path in best_matches:
        print(f"📸 {img_path} - Similarity Score: {abs(score):.3f}")


    # Hiển thị ảnh mới
    plt.imshow(cv2.cvtColor(cv2.imread(img_new_path), cv2.COLOR_BGR2RGB))
    plt.title("Ảnh đã tải lên")
    plt.show()

    # Hiển thị 5 ảnh tương đồng nhất
    no = 1
    for score, path in best_matches:
        img = cv2.imread(path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Ứng cử viên số {no} \nScore Different: {abs(score):.2f}")
        plt.show()
        no += 1
