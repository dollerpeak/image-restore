import cv2
import numpy as np
import os
import random
from glob import glob
from tqdm import tqdm

# 원본 이미지가 없고
# grid형식으로 조각난 이미지가 있고
# row, cols 값을 알고
# 조각난 이미지 순서는 모르고 파일명은 랜덤인 경우

# --------------------
# 환경 설정
original_pieces_path = "D:/workspace-python/image-restore/pieces"  # 조각 이미지 폴더 경로 여기에 입력
rows = 5  # 행 개수
cols = 7  # 열 개수
# --------------------

def normalize_brightness(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)
    img_lab = cv2.merge((l,a,b))
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

def load_and_preprocess_pieces(folder_path):
    files = glob(os.path.join(folder_path, "*"))
    images = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        img = normalize_brightness(img)
        images.append(img)
    return images

def resize_pieces(images, size):
    resized = [cv2.resize(im, size) for im in images]
    return resized

def get_orb_features(img):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return len(matches)

def extract_edge_strip(img, side, strip_width=5):
    # side: 'left', 'right', 'top', 'bottom'
    if side == 'left':
        return img[:, :strip_width]
    elif side == 'right':
        return img[:, -strip_width:]
    elif side == 'top':
        return img[:strip_width, :]
    elif side == 'bottom':
        return img[-strip_width:, :]

def edge_similarity(img1, side1, img2, side2):
    strip1 = extract_edge_strip(img1, side1)
    strip2 = extract_edge_strip(img2, side2)
    # RGB 차이 평균 (노멀라이즈)
    color_diff = np.mean(np.abs(strip1.astype(np.float32) - strip2.astype(np.float32)))
    # 특징점 매칭 점수 (ORB)
    _, desc1 = get_orb_features(strip1)
    _, desc2 = get_orb_features(strip2)
    feature_score = match_features(desc1, desc2)
    # 점수 결합 (가중치 설정 가능)
    # 특징점 매칭이 많을수록 similarity 높다고 가정
    # 따라서 색상 차이와 특징점 매칭을 조합할 때, 특징점 매칭은 음의 값으로 반영
    # 점수는 작을수록 유사 (color_diff 낮고 feature_score 높음)
    similarity = color_diff - feature_score * 5  # 특징점에 가중치 5 적용
    return similarity

def build_similarity_matrix(images):
    N = len(images)
    similarity = np.full((N, N, 4), np.inf, dtype=np.float32)  # 4방향 (right, bottom)
    # 저장: similarity[i,j,0] = i의 오른쪽 엣지와 j의 왼쪽 엣지 차이
    # similarity[i,j,1] = i의 아래쪽 엣지와 j의 위쪽 엣지 차이

    for i in tqdm(range(N)):
        for j in range(N):
            if i == j:
                continue
            similarity[i,j,0] = edge_similarity(images[i], 'right', images[j], 'left')
            similarity[i,j,1] = edge_similarity(images[i], 'bottom', images[j], 'top')
    return similarity

def initial_heuristic_placement(similarity, N, rows, cols):
    # 간단한 휴리스틱: similarity에서 가장 낮은 값부터 연결 시도
    # 조각 하나를 (0,0)에 놓고, 오른쪽과 아래로 차례대로 최소 similarity 조각 배치
    placement = -np.ones((rows, cols), dtype=int)
    used = set()

    placement[0,0] = 0
    used.add(0)

    for r in range(rows):
        for c in range(cols):
            if placement[r,c] != -1:
                continue
            candidates = []
            # 왼쪽 조각과 오른쪽 엣지 비교
            if c > 0 and placement[r,c-1] != -1:
                left_idx = placement[r,c-1]
                for i in range(N):
                    if i in used:
                        continue
                    candidates.append((similarity[left_idx,i,0], i))
            # 위쪽 조각과 아래 엣지 비교
            if r > 0 and placement[r-1,c] != -1:
                top_idx = placement[r-1,c]
                for i in range(N):
                    if i in used:
                        continue
                    candidates.append((similarity[top_idx,i,1], i))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                for sim_score, cand_idx in candidates:
                    if cand_idx not in used:
                        placement[r,c] = cand_idx
                        used.add(cand_idx)
                        break
            else:
                # 후보 없으면 랜덤 선택
                for i in range(N):
                    if i not in used:
                        placement[r,c] = i
                        used.add(i)
                        break
    return placement

def calculate_total_score(placement, similarity):
    score = 0
    rows, cols = placement.shape
    for r in range(rows):
        for c in range(cols):
            idx = placement[r,c]
            if c < cols - 1:
                right_idx = placement[r,c+1]
                score += similarity[idx,right_idx,0]
            if r < rows -1:
                bottom_idx = placement[r+1,c]
                score += similarity[idx,bottom_idx,1]
    return score

def simulated_annealing(placement, similarity, max_iter=10000, temp_init=1000, temp_final=1, alpha=0.995):
    rows, cols = placement.shape
    current_score = calculate_total_score(placement, similarity)
    current_placement = placement.copy()
    best_placement = placement.copy()
    best_score = current_score
    temp = temp_init

    for it in tqdm(range(max_iter)):
        if temp < temp_final:
            break
        # 두 조각 위치 스왑
        r1, c1 = random.randint(0, rows-1), random.randint(0, cols-1)
        r2, c2 = random.randint(0, rows-1), random.randint(0, cols-1)
        if (r1, c1) == (r2, c2):
            continue

        new_placement = current_placement.copy()
        new_placement[r1,c1], new_placement[r2,c2] = new_placement[r2,c2], new_placement[r1,c1]
        new_score = calculate_total_score(new_placement, similarity)
        delta = new_score - current_score

        if delta < 0 or random.random() < np.exp(-delta/temp):
            current_placement = new_placement
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_placement = current_placement

        temp *= alpha

    return best_placement

def stitch_images(placement, images):
    rows, cols = placement.shape
    h, w = images[0].shape[:2]
    final_img = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            idx = placement[r,c]
            final_img[r*h:(r+1)*h, c*w:(c+1)*w] = images[idx]
    return final_img

def main():
    print("1. 이미지 불러오기 및 전처리")
    pieces = load_and_preprocess_pieces(original_pieces_path)
    if len(pieces) != rows * cols:
        print(f"조각 개수와 rows*cols({rows*cols})가 다릅니다. 실제 조각 수: {len(pieces)}")
        return
    resize_shape = (pieces[0].shape[1], pieces[0].shape[0])  # (width, height)
    pieces = resize_pieces(pieces, resize_shape)

    print("2. 조각 간 유사도 행렬 계산 (시간 소요 예상)")
    similarity = build_similarity_matrix(pieces)

    print("3. 초기 휴리스틱 배치")
    placement = initial_heuristic_placement(similarity, len(pieces), rows, cols)

    print("4. 전역 탐색 (Simulated Annealing) 시작")
    placement = simulated_annealing(placement, similarity)

    print("5. 결과 이미지 생성 및 저장")
    result = stitch_images(placement, pieces)
    cv2.imwrite("restored_image.jpg", result)
    print("복원 이미지 저장 완료: restored_image.jpg")

if __name__ == "__main__":
    main()







