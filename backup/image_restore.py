import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from itertools import product
import math

# 설정
PIECES_DIR = "pieces"
ROTATION_ANGLES = [0, 90, 180, 270]
SSIM_THRESHOLD = 0.6

def load_pieces(path=PIECES_DIR):
    images = []
    for filename in sorted(os.listdir(path)):
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath)
        if img is not None:
            images.append((filename, img))
    return images

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    return rotated

def ssim_score(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h = min(img1_gray.shape[0], img2_gray.shape[0])
    w = min(img1_gray.shape[1], img2_gray.shape[1])
    img1_crop = cv2.resize(img1_gray, (w, h))
    img2_crop = cv2.resize(img2_gray, (w, h))
    score = ssim(img1_crop, img2_crop)
    return score

def generate_rotated_variants(pieces):
    rotated = []
    for name, img in pieces:
        variants = [(angle, rotate_image(img, angle)) for angle in ROTATION_ANGLES]
        rotated.append((name, variants))
    return rotated

def compute_similarity_matrix(rotated_pieces):
    n = len(rotated_pieces)
    matrix = [[0.0]*n for _ in range(n)]
    for i, j in product(range(n), repeat=2):
        if i == j:
            continue
        max_score = 0
        for angle1, img1 in rotated_pieces[i][1]:
            for angle2, img2 in rotated_pieces[j][1]:
                score = ssim_score(img1, img2)
                if score > max_score:
                    max_score = score
        matrix[i][j] = max_score
    return matrix

def assemble_image(rotated_pieces, grid_size=(3, 3)):
    pw, ph = 0, 0
    for _, variants in rotated_pieces:
        img = variants[0][1]
        h, w = img.shape[:2]
        pw = max(pw, w)
        ph = max(ph, h)

    final_h = ph * grid_size[0]
    final_w = pw * grid_size[1]
    final_img = np.full((final_h, final_w, 3), (0, 0, 255), dtype=np.uint8)  # 빨간색으로 초기화

    i = 0
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            if i < len(rotated_pieces):
                img = rotated_pieces[i][1][0][1]
                img_resized = cv2.resize(img, (pw, ph))
                final_img[row*ph:(row+1)*ph, col*pw:(col+1)*pw] = img_resized
            i += 1
    return final_img

# 실행
if __name__ == "__main__":
    pieces = load_pieces()
    rotated_variants = generate_rotated_variants(pieces)
    similarity_matrix = compute_similarity_matrix(rotated_variants)

    print("SSIM 유사도 행렬:")
    for row in similarity_matrix:
        print(["{:.2f}".format(v) for v in row])

    # 그리드 사이즈 수동 조정 (예: 3행 x 3열)
    # grid_rows, grid_cols = 3, 3
    num_pieces = len(rotated_variants)
    grid_cols = math.ceil(math.sqrt(num_pieces))
    grid_rows = math.ceil(num_pieces / grid_cols)
    result_img = assemble_image(rotated_variants, grid_size=(grid_rows, grid_cols))

    cv2.imwrite("restored_result.jpg", result_img)
    print("복원된 이미지: restored_result.jpg 저장 완료")
