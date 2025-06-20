import cv2
import numpy as np
import os
import math

# 이미지 크기 맞추기 (자동으로 가장 작은 크기에 맞춤)
def resize_pieces(images, target_size=None):
    if target_size is None:
        h_min = min(img.shape[0] for img in images)
        w_min = min(img.shape[1] for img in images)
        target_size = (w_min, h_min)

    resized = [cv2.resize(img, target_size) for img in images]
    return resized

# 유사도 측정 (가장자리 색 차이)
def edge_diff(left, right):
    return np.sum(cv2.absdiff(left[:, -5:], right[:, :5]))

# 이미지 조각 로드
def load_pieces(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            images.append(img)
    return images

# 자동으로 그리드 크기 추정 (가로:열, 세로:행)
def estimate_grid_size(num_pieces):
    sqrt_n = int(math.sqrt(num_pieces))
    for i in range(sqrt_n, 0, -1):
        if num_pieces % i == 0:
            return (num_pieces // i, i)  # (cols, rows)
    return (num_pieces, 1)  # fallback

# 조각 정렬 (좌우로만 유사도 기준, 간단한 프로토타입)
def arrange_pieces(pieces, grid_cols, grid_rows):
    used = [False] * len(pieces)
    arranged = []

    # 첫 행 만들기
    row = [0]
    used[0] = True
    for _ in range(1, grid_cols):
        last = pieces[row[-1]]
        best = -1
        best_score = float('inf')
        for i, p in enumerate(pieces):
            if not used[i]:
                score = edge_diff(last, p)
                if score < best_score:
                    best_score = score
                    best = i
        row.append(best)
        used[best] = True
    arranged.append(row)

    # 나머지 행은 랜덤하게 (확장 가능)
    remaining = [i for i, u in enumerate(used) if not u]
    for _ in range(1, grid_rows):
        line = remaining[:grid_cols]
        remaining = remaining[grid_cols:]
        arranged.append(line)

    return arranged

# 조각 병합
def merge_grid(pieces, arrangement):
    rows = []
    for row in arrangement:
        imgs = [pieces[i] for i in row]
        rows.append(cv2.hconcat(imgs))
    final = cv2.vconcat(rows)
    return final

# 메인 함수
# def restore_image(folder="pieces", output="restored.jpg"):
def restore_image(folder="pieces", output="restored.jpg"):
    raw_pieces = load_pieces(folder)
    if len(raw_pieces) < 2:
        print("조각 이미지가 2개 이상 필요합니다.")
        return

    resized_pieces = resize_pieces(raw_pieces)
    cols, rows = estimate_grid_size(len(resized_pieces))
    print(f"그리드 크기: {cols} x {rows}")

    arrangement = arrange_pieces(resized_pieces, cols, rows)
    result = merge_grid(resized_pieces, arrangement)
    cv2.imwrite(output, result)
    print(f"복원된 이미지를 '{output}'로 저장했습니다.")

if __name__ == "__main__":
    restore_image()
