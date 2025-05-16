import numpy as np
import psycopg2
import psycopg2.extras
import random
import json

from tqdm import tqdm

from datetime import datetime
# Kết nối PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname="recognize_office_equipment", user="postgres", password="17082002", host="localhost"
    )


# Tải dataset từ PostgreSQL
def load_dataset():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT feature_vector, label FROM image_features")
    dataset = [(np.array(row[0]), row[1]) for row in cursor.fetchall()]
    conn.close()
    return dataset

# Khởi tạo quần thể
num_features = 1044  # Giả sử mỗi vector có 256 đặc trưng
num_individuals = 20  # Giảm số cá thể trong quần thể
num_generations = 50  # Giảm số thế hệ

def initialize_population():
    population = np.random.rand(num_individuals, num_features)
    population /= population.sum(axis=1, keepdims=True)  # Chuẩn hóa tổng w = 1
    return population

# Hàm đánh giá độ phù hợp
def fitness(w, dataset, sample_size):
    score = 0
    # Lấy mẫu ngẫu nhiên từ dataset
    sample = random.sample(dataset, sample_size)
    for img1, label1 in sample:
        for img2, label2 in sample:
            if not np.array_equal(img1, img2):
                # d = np.sum(w * np.abs(img1 - img2))
                d = np.sum(w * (np.abs(img1 - img2) / (np.abs(img1) + np.abs(img2) + 1e-8)))
                score -= d if label1 == label2 else -d
                # if label1 == label2:
                #     score -= d  # Giữ nguyên
                # else:
                #     score += d  # Đổi dấu để đảm bảo tối đa hóa khoảng cách

    return score

# Chọn lọc tự nhiên
def select_best(population, fitness_scores, num_best=7):
    indices = np.argsort(fitness_scores)[-num_best:]
    return population[indices]

# Lai ghép
def crossover(parent1, parent2):
    alpha = np.random.rand(len(parent1))
    child = alpha * parent1 + (1 - alpha) * parent2
    return child / np.sum(child)

# Đột biến
def mutate(child, mutation_rate=0.2):
    mutation = np.random.randn(len(child)) * mutation_rate
    child += mutation
    child = np.clip(child, 0, 1)
    return child / np.sum(child)

# Chạy Genetic Algorithm
def genetic_algorithm(num_generations=num_generations):
    dataset = load_dataset()
    population = initialize_population()

    # dataset = random.sample(dataset, 150) # fixed để tránh bị biến động fitness_score dù cùng một cá thế W(w1, w2, w3, ...)

    # In thông tin dataset để theo dõi
    print(f"Dataset has {len(dataset)} samples.")

    print("\n🚀 Bắt đầu tối ưu hóa bằng Genetic Algorithm...\n")
    
    best_fitness = -np.inf  # Biến lưu trữ fitness tốt nhất
    best_population = None  # Biến lưu trữ quần thể tốt nhất

    for generation in range(num_generations):
        start_gen_time = datetime.now()

        # Tính toán fitness scores cho quần thể hiện tại
        fitness_scores = np.array([fitness(w, dataset, 100) for w in tqdm(population, desc=f"🔄 Generation {generation + 1}")])
        
        # Kiểm tra nếu fitness_scores có phần tử hợp lệ
        if fitness_scores.size == 0:
            print(f"⚠️ Warning: Fitness scores are empty for Generation {generation}. Skipping generation.")
            continue

        # Chọn quần thể tốt nhất
        best_population = select_best(population, fitness_scores)

        # 🔥 Giữ lại cá thể ưu tú nhất
        elite = population[np.argmax(fitness_scores)]  

        # Lai ghép và đột biến để tạo ra quần thể mới
        new_population = []
        for _ in range(len(population) // 2):
            p1, p2 = random.sample(list(best_population), 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        
        # Đảm bảo rằng số lượng quần thể sau khi lai ghép và đột biến đủ bằng số lượng quần thể ban đầu
        while len(new_population) < len(population) - 1:
            # Tạo thêm cá thể từ quần thể tốt nhất nếu cần thiết (kiểu như padding)
            new_population.append(random.choice(best_population))

        new_population.append(elite)
        population = np.array(new_population)

        # Ghi lại thời gian kết thúc thế hệ
        end_gen_time = datetime.now()
        gen_duration = end_gen_time - start_gen_time

        # Hiển thị thông tin chi tiết về thế hệ hiện tại
        print(f"📌 Generation {generation + 1}:")
        print(f"   🏆 Best fitness  = {np.max(fitness_scores):.2f}")
        print(f"   📊 Avg fitness   = {np.mean(fitness_scores):.2f}")
        print(f"   Time taken    = {str(gen_duration).split('.')[0]}\n")

        # Cập nhật best_fitness và best_population nếu tìm được cá thể tốt hơn
        current_best_fitness = np.max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_population = population[np.argmax(fitness_scores)]

    # Trả về cá thể tốt nhất
    if best_population is not None:
        return population[np.argmax(fitness_scores)] # best_population
    else:
        print("⚠️ No valid fitness scores were computed across all generations. Unable to return the best population.")
        return None

# Lưu trọng số tối ưu vào PostgreSQL
def save_best_weights(best_w):
    conn = connect_db()
    cursor = conn.cursor()

    # Chuyển đổi mảng NumPy thành danh sách Python
    best_w_list = best_w.tolist()

    # Lưu trữ mảng vào cơ sở dữ liệu PostgreSQL với kiểu REAL[]
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimal_weights (
            id SERIAL PRIMARY KEY,
            weights REAL[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Thực thi câu lệnh INSERT để lưu trữ mảng REAL[] vào bảng
    cursor.execute("INSERT INTO optimal_weights (weights) VALUES (%s)", (best_w_list,))
    conn.commit()
    conn.close()

if __name__ == "__main__": # Mọi đoạn mã không được bảo vệ bởi "__main__" sẽ được chạy ngay khi file chính được import.
    # Chạy tối ưu hóa
    start_time = datetime.now()

    best_w = genetic_algorithm()
    if best_w is not None:
        save_best_weights(best_w)
    else:
        print("⚠️ No valid best weights found, skipping save.")

    end_time = datetime.now()
    print(f"✅✅✅ Hoàn thành tối ưu trọng số w và lưu vào PostgreSQL.\n- Start time = {start_time.strftime('%H:%M:%S %d-%m-%Y')} \n- End time = {end_time.strftime('%H:%M:%S %d-%m-%Y')}.\n- Total = {str(end_time - start_time).split('.')[0]}s." )
# print(f"=== OPTIMIZATION FINISHED ===")
# print(f"|| Bắt đầu lúc    : {start_time.strftime('%H:%M:%S')}||")
# print(f"|| Kết thúc lúc   : {end_time.strftime('%H:%M:%S')}||")
# print(f"|| Tổng thời gian : {end_time - start_time}s||")
# print(f"==== DAY TEST {datetime.now().strftime('%d-%m-%Y')} ====")

# Chẳng hạn chạy 50 thế hệ - num_generations = 50, với số cá thể là 20 - num_individuals = 20
# Khi đó kết quả sẽ là
# bảng Optimal_weights
# ID 1
# weight = 
# {
#     {},
#     {},
#     ...,
#     {} [50 cá thể - bộ trọng số w]
# }