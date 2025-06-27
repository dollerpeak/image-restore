import cv2
import os
import shutil
import random
import string

def clear_pieces_folder(folder='pieces'):
    print(f"{folder} 디렉토리 파일 전체 삭제")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def random_filename(length=8, ext='png'):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choices(letters, k=length)) + f'.{ext}'

def slice_image_grid(image_path, rows, cols, save_folder='pieces'):
    # 1. 폴더 초기화
    clear_pieces_folder(save_folder)

    # 2. 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    height, width = img.shape[:2]
    piece_h = height // rows
    piece_w = width // cols

    print(f"이미지 크기: {width}x{height}, 조각 크기: {piece_w}x{piece_h}")

    # 3. 조각 자르기 및 저장
    piece_paths = []
    for r in range(rows):
        for c in range(cols):
            # 자르기 영역
            y0 = r * piece_h
            x0 = c * piece_w
            # 마지막 행,열은 나머지 포함
            y1 = (r + 1) * piece_h if r < rows - 1 else height
            x1 = (c + 1) * piece_w if c < cols - 1 else width

            piece = img[y0:y1, x0:x1]

            # 임의 파일명 생성 후 저장
            fname = random_filename()
            fpath = os.path.join(save_folder, fname)
            cv2.imwrite(fpath, piece)
            piece_paths.append(fpath)

    print(f"{len(piece_paths)}개의 조각 저장 완료 (폴더: {save_folder})")
    return piece_paths


if __name__ == '__main__':
    # 테스트용: 이미지 경로, 행, 열 지정
    IMAGE_PATH = 'test_3.jpg'  # 원본 이미지 파일명 바꿔서 사용하세요
    ROWS = 5
    COLS = 7
    print(f"파일명 = {IMAGE_PATH}, rows = {ROWS} * cols = {COLS}")

    slice_image_grid(IMAGE_PATH, ROWS, COLS)


