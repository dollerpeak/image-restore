import cv2
import numpy as np
import os

# 원본 이미지가 있고
# grid형식으로 조각난 이미지가 있고
# row, cols 값을 모르고
# 조각난 이미지 순서는 모르고 파일명은 랜덤인 경우

# ====== 설정 부분 =======
original_image_path = "D:/workspace-python/image-restore/test_1.jpg"  # 원본 이미지 경로 작성
pieces_folder_path = "D:/workspace-python/image-restore/pieces"        # 조각 이미지 폴더 경로 작성
# ========================

def load_pieces(folder_path):
    pieces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            pieces.append(img)
    return pieces

def guess_grid_size(original_img, pieces):
    orig_h, orig_w = original_img.shape[:2]

    # 조각 이미지 크기를 모두 확인해서 가장 작은 크기를 선택 (조각 크기가 같다고 가정)
    piece_heights = [p.shape[0] for p in pieces]
    piece_widths = [p.shape[1] for p in pieces]
    piece_h = min(piece_heights)
    piece_w = min(piece_widths)

    # 원본 이미지 크기에서 조각 크기를 나누어 rows, cols 추정
    rows = orig_h // piece_h
    cols = orig_w // piece_w

    return rows, cols, piece_h, piece_w

def split_original(original_img, rows, cols):
    h, w, _ = original_img.shape
    piece_h = h // rows
    piece_w = w // cols
    pieces = []
    for r in range(rows):
        for c in range(cols):
            piece = original_img[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w]
            pieces.append(piece)
    return pieces

def compare_images(img1, img2):
    # 크기 맞춤
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return diff

def find_matches(original_pieces, shuffled_pieces):
    matches = [-1]*len(original_pieces)
    used = set()
    for i, orig_piece in enumerate(original_pieces):
        min_diff = float('inf')
        min_idx = -1
        for j, shuf_piece in enumerate(shuffled_pieces):
            if j in used:
                continue
            diff = compare_images(orig_piece, shuf_piece)
            if diff < min_diff:
                min_diff = diff
                min_idx = j
        matches[i] = min_idx
        used.add(min_idx)
    return matches

def reconstruct_image(shuffled_pieces, matches, rows, cols, piece_h, piece_w):
    full_img = np.zeros((piece_h * rows, piece_w * cols, 3), dtype=np.uint8)
    for idx_orig, idx_piece in enumerate(matches):
        r = idx_orig // cols
        c = idx_orig % cols
        piece_img = shuffled_pieces[idx_piece]
        piece_img = cv2.resize(piece_img, (piece_w, piece_h))
        full_img[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w] = piece_img
    return full_img

if __name__ == "__main__":
    original_img = cv2.imread(original_image_path)
    shuffled_pieces = load_pieces(pieces_folder_path)

    rows, cols, piece_h, piece_w = guess_grid_size(original_img, shuffled_pieces)
    print(f"추정된 grid size: rows={rows}, cols={cols}, piece size=({piece_h},{piece_w})")

    original_pieces = split_original(original_img, rows, cols)
    matches = find_matches(original_pieces, shuffled_pieces)

    reconstructed = reconstruct_image(shuffled_pieces, matches, rows, cols, piece_h, piece_w)
    cv2.imwrite("reconstructed.png", reconstructed)
    print("복원된 이미지가 reconstructed.png 파일로 저장되었습니다.")


