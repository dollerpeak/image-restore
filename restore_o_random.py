import cv2
import numpy as np
import os

# 원본 이미지가 있고
# 랜덤하게 조각난 이미지가 있고 (gird형식으로 조각난게 아니라 이미지마다 크기 틀림)
# 조각난 이미지 순서는 모르고 파일명은 랜덤인 경우

# ====== 설정 ======
original_image_path = "D:/workspace-python/image-restore/test_1.jpg"  # 원본 이미지 경로 작성
pieces_folder_path = "D:/workspace-python/image-restore/pieces"        # 조각 이미지 폴더 경로 작성
output_path = "reconstructed.png"                   # 출력 파일명
# ===================

def load_piece_images(folder_path):
    pieces = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path)
            if img is not None:
                pieces.append((fname, img))
    return pieces

def match_and_paste(original_img, pieces):
    reconstructed = original_img.copy()
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)

    for fname, piece in pieces:
        print(f"Matching piece: {fname}")

        # 원본보다 조각이 크면 무시
        if piece.shape[0] > original_img.shape[0] or piece.shape[1] > original_img.shape[1]:
            print(f"  Skipped (too large): {piece.shape}")
            continue

        # 템플릿 매칭 (grayscale)
        res = cv2.matchTemplate(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY),
                                cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(f"  Best match at {max_loc}, score={max_val:.3f}")

        top_left = max_loc
        h, w = piece.shape[:2]

        # 조각 덮어쓰기
        reconstructed[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w] = piece

        # (선택) 시각적 디버그용 마스크 표시
        cv2.rectangle(mask, top_left, (top_left[0]+w, top_left[1]+h), 255, -1)

    return reconstructed, mask

if __name__ == "__main__":
    original = cv2.imread(original_image_path)
    if original is None:
        raise Exception("원본 이미지를 불러올 수 없습니다.")

    pieces = load_piece_images(pieces_folder_path)
    if not pieces:
        raise Exception("조각 이미지를 찾을 수 없습니다.")

    result, coverage_mask = match_and_paste(original, pieces)
    cv2.imwrite(output_path, result)
    cv2.imwrite("coverage_mask.png", coverage_mask)
    print(f"복원 완료: {output_path}")
