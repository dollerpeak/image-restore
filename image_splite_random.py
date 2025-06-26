import cv2
import numpy as np
import random
import os
import shutil

def random_partition(total_length, parts):
    """total_length를 parts 개로 랜덤 분할 (합은 total_length)"""
    if parts == 1:
        return [total_length]
    cuts = sorted(random.sample(range(1, total_length), parts - 1))
    sizes = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [total_length - cuts[-1]]
    return sizes

def split_image_varied_grid(image, num_pieces):
    """이미지를 num_pieces 개로 분할 (누락/중복 없음), 가변 크기 조각"""
    h, w, _ = image.shape

    # 행, 열 개수 결정: num_pieces를 최대한 정사각형에 가깝게 분할
    best_diff = float('inf')
    best_r, best_c = 1, num_pieces
    for r in range(1, num_pieces + 1):
        c = (num_pieces + r - 1) // r  # 올림
        diff = abs(r - c)
        if diff < best_diff:
            best_diff = diff
            best_r, best_c = r, c

    # 행과 열 크기 난수 분할
    row_sizes = random_partition(h, best_r)
    col_sizes = random_partition(w, best_c)

    pieces = []
    y = 0
    count = 0
    for rh in row_sizes:
        x = 0
        for cw in col_sizes:
            if count >= num_pieces:
                break
            piece = image[y:y + rh, x:x + cw].copy()
            pieces.append(piece)
            x += cw
            count += 1
        y += rh

    return pieces

def clear_pieces_dir(pieces_dir):
    if os.path.exists(pieces_dir):
        shutil.rmtree(pieces_dir)
    os.makedirs(pieces_dir)

def save_pieces(pieces, pieces_dir):
    for i, piece in enumerate(pieces):
        filename = f"piece_{i + 1}.png"
        filepath = os.path.join(pieces_dir, filename)
        cv2.imwrite(filepath, piece)

def main():
    # --- 수정 부분: 여기서 이미지 파일명과 분할 조각 수 설정 ---
    input_image_path = "test_1.jpg"  # 입력 이미지 파일 경로 (여기서 바꾸세요)
    num_pieces = 12                # 조각 수 (여기서 바꾸세요)

    pieces_dir = "pieces"
    clear_pieces_dir(pieces_dir)

    image = cv2.imread(input_image_path)
    if image is None:
        print(f"이미지 파일을 읽을 수 없습니다: {input_image_path}")
        return

    pieces = split_image_varied_grid(image, num_pieces)
    save_pieces(pieces, pieces_dir)

    print(f"{num_pieces}개 조각을 '{pieces_dir}' 폴더에 저장했습니다.")

if __name__ == "__main__":
    main()
