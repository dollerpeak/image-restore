import os
import time
from tkinter import ttk
import cv2
import uuid
import shutil
import random
import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

original_image = None
original_image_path = ''
pieces_dir = os.path.join(os.getcwd(), 'pieces')
restore_dir = os.path.join(os.getcwd(), 'restore')
preview_window = None
pieces_preview_window = None
vis_window = None


def load_pieces(directory):
    """pieces 디렉토리에서 이미지 파일들을 읽어와 리스트로 반환"""
    pieces = []
    if not os.path.exists(directory):
        print(f"조각 디렉토리 없음: {directory}")
        return pieces

    files = sorted(os.listdir(directory))
    for fname in files:
        path = os.path.join(directory, fname)
        if fname.lower().endswith(('.jpg', '.png')):
            img = cv2.imread(path)
            if img is not None:
                pieces.append((fname, img))
    return pieces


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def is_allowed_file(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in ['.png', '.jpg', '.jpeg']


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def show_image_preview(img_path):
    global preview_window
    if preview_window and preview_window.winfo_exists():
        preview_window.destroy()
    preview_window = tk.Toplevel()
    preview_window.title("원본 이미지 미리보기")
    screen_w, screen_h = get_screen_size()
    img = Image.open(img_path)
    img_w, img_h = img.size
    max_w = screen_w // 3 if img_w > screen_w else img_w
    max_h = screen_h // 3 if img_h > screen_h else img_h
    img.thumbnail((max_w, max_h))
    preview_window.geometry(f"{img.width}x{img.height}")
    img_tk = ImageTk.PhotoImage(img)
    label = tk.Label(preview_window, image=img_tk)
    label.image = img_tk
    label.pack()


def show_pieces_preview(image_paths):
    global pieces_preview_window
    if pieces_preview_window and pieces_preview_window.winfo_exists():
        pieces_preview_window.destroy()
    pieces_preview_window = tk.Toplevel()
    pieces_preview_window.title("조각 이미지 미리보기")
    max_width = max(Image.open(p).width for p in image_paths)
    screen_w, screen_h = get_screen_size()
    win_h = screen_h // 2
    win_w = min(max_width + 60, screen_w // 2)
    pieces_preview_window.geometry(f"{win_w}x{win_h}")
    container = tk.Frame(pieces_preview_window)
    container.pack(fill="both", expand=True)
    canvas = tk.Canvas(container, width=win_w, height=win_h)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollable_frame = tk.Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    scrollable_frame.bind_all("<MouseWheel>", _on_mousewheel)
    for img_path in sorted(image_paths):
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        label = tk.Label(scrollable_frame, image=img_tk)
        label.image = img_tk
        label.pack(pady=5)


def load_original_image():
    global original_image, original_image_path, preview_window, pieces_preview_window, vis_window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if not file_path:
        return
    if not is_allowed_file(file_path):
        messagebox.showerror("오류", "허용된 확장자는 png, jpg, jpeg 입니다.")
        return
    original_image = cv2.imread(file_path)
    original_image_path = file_path
    show_image_preview(file_path)
    for win in [pieces_preview_window, vis_window]:
        if win and win.winfo_exists():
            win.destroy()
    clear_directory(pieces_dir)
    clear_directory(restore_dir)


def split_image_grid(img, rows, cols, ext):
    h, w, _ = img.shape
    piece_h, piece_w = h // rows, w // cols
    for r in range(rows):
        for c in range(cols):
            piece = img[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w]
            filename = os.path.join(pieces_dir, f"{uuid.uuid4()}{ext}")
            cv2.imwrite(filename, piece)


# def split_image_random(img, num_pieces, ext):
def split_image_random(img, ext):
    pieces_dir = "pieces"
    if os.path.exists(pieces_dir):
        shutil.rmtree(pieces_dir)
    os.makedirs(pieces_dir)

    height, width = img.shape[:2]
    rows = random.randint(10, 20)
    cols = random.randint(10, 20)
    cell_h = height // rows
    cell_w = width // cols

    used = np.zeros((rows, cols), dtype=bool)
    idx = 0

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and not used[r, c]

    for r in range(rows):
        for c in range(cols):
            if used[r, c]:
                continue

            # 병합 후보 셀 (자신 포함)
            group = [(r, c)]
            used[r, c] = True

            max_merge = random.randint(2, 5)
            directions = [(-1,0),(1,0),(0,-1),(0,1)]  # 상하좌우
            while len(group) < max_merge:
                possible = []
                for gr, gc in group:
                    for dr, dc in directions:
                        nr, nc = gr + dr, gc + dc
                        if is_valid(nr, nc) and (nr, nc) not in group:
                            possible.append((nr, nc))
                if not possible:
                    break
                choice = random.choice(possible)
                group.append(choice)
                used[choice[0], choice[1]] = True

            # 병합된 영역 추출
            min_r = min(g[0] for g in group)
            max_r = max(g[0] for g in group)
            min_c = min(g[1] for g in group)
            max_c = max(g[1] for g in group)

            y1 = min_r * cell_h
            y2 = (max_r + 1) * cell_h if max_r + 1 < rows else height
            x1 = min_c * cell_w
            x2 = (max_c + 1) * cell_w if max_c + 1 < cols else width

            piece = img[y1:y2, x1:x2]
            filename = os.path.join(pieces_dir, f"{uuid.uuid4()}{ext}")
            cv2.imwrite(filename, piece)
            idx += 1

    print(f"[INFO] 총 {idx}개의 조각을 생성했습니다. 겹침/누락 없음.")


def perform_split():
    global vis_window
    if original_image is None:
        messagebox.showwarning("경고", "먼저 원본 이미지를 불러와 주세요.")
        return

    if vis_window and vis_window.winfo_exists():
        vis_window.destroy()

    clear_directory(pieces_dir)
    ext = os.path.splitext(original_image_path)[-1].lower()
    method = split_method.get()

    if method == "grid":
        try:
            rows = int(entry_rows.get())
            cols = int(entry_cols.get())
        except ValueError:
            messagebox.showerror("오류", "행과 열은 숫자로 입력해야 합니다.")
            return

        # 범위 제한 및 자동 보정
        adjusted = False
        if rows < 1:
            rows = 1
            adjusted = True
        elif rows > 10:
            rows = 10
            adjusted = True

        if cols < 1:
            cols = 1
            adjusted = True
        elif cols > 10:
            cols = 10
            adjusted = True

        if adjusted:
            messagebox.showinfo("알림", "행과 열 값은 1~10 사이로 자동 조정되었습니다.")
            entry_rows.delete(0, tk.END)
            entry_rows.insert(0, str(rows))
            entry_cols.delete(0, tk.END)
            entry_cols.insert(0, str(cols))

        split_image_grid(original_image, rows, cols, ext)
    else:
        split_image_random(original_image, ext)
    piece_files = [os.path.join(pieces_dir, f) for f in sorted(os.listdir(pieces_dir)) if is_allowed_file(f)]
    show_pieces_preview(piece_files)
    messagebox.showinfo("완료", f"{len(piece_files)}개의 이미지 조각을 생성했습니다.")
    

def match_and_paste(original_img, pieces, vis_label, update_display):
    reconstructed = np.zeros_like(original_img)
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

        # 실시간 UI 업데이트
        vis_rgb = cv2.cvtColor(reconstructed.copy(), cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(vis_rgb)
        img_pil.thumbnail((vis_label.winfo_width(), vis_label.winfo_height()))
        img_tk = ImageTk.PhotoImage(img_pil)
        vis_label.configure(image=img_tk)
        vis_label.image = img_tk
        vis_label.update()
        time.sleep(0.5)

        # 디버그용 마스크 표시
        cv2.rectangle(mask, top_left, (top_left[0]+w, top_left[1]+h), 255, -1)

    return reconstructed, mask


def match_template_piece(original_img, piece, canvas):
    """
    조각 이미지를 원본 이미지에서 가장 유사한 위치에 복사해서 붙임
    - original_img: 원본 이미지
    - piece: 조각 이미지
    - canvas: 붙여넣기용 이미지 (복사본)
    """
    res = cv2.matchTemplate(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    h, w = piece.shape[:2]
    canvas[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w] = piece


def restore_images():
    def update_display(image, label, w, h):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((w, h))
        img_tk = ImageTk.PhotoImage(img)
        label.configure(image=img_tk)
        label.image = img_tk
        label.update()

    if not original_image_path or not os.path.exists(original_image_path):
        messagebox.showwarning("경고", "원본 이미지를 먼저 불러와주세요.")
        return

    pieces = load_pieces(pieces_dir)
    if not pieces:
        messagebox.showwarning("경고", "조각 이미지가 없습니다.")
        return
    
    global vis_window
    if vis_window and vis_window.winfo_exists():
        vis_window.destroy()

    original_image = cv2.imread(original_image_path)
    restore_dir = "restore"
    os.makedirs(restore_dir, exist_ok=True)
    
    vis_window = tk.Toplevel()
    vis_window.title("복원 이미지 (실시간)")
    height, width = original_image.shape[:2]
    vis_window.geometry(f"{width}x{height}")
    vis_label = ttk.Label(vis_window)
    vis_label.pack(fill=tk.BOTH, expand=True)
    vis_window.update()
    win_w, win_h = vis_label.winfo_width(), vis_label.winfo_height()

    # 조각 타입 판별
    sizes = [p.shape[:2] for _, p in pieces if p is not None]
    heights, widths = zip(*sizes)
    max_h, min_h = max(heights), min(heights)
    max_w, min_w = max(widths), min(widths)
    is_grid = (max_h - min_h <= 2) and (max_w - min_w <= 2)

    if is_grid:
        print("Grid 방식 감지됨 → 기존 방식으로 복원")
        restored = np.zeros_like(original_image)

        def process_piece(index):
            if index >= len(pieces):
                cv2.imwrite(os.path.join(restore_dir, "restored.jpg"), restored)
                messagebox.showinfo("완료", "복원이 완료되었습니다.")
                return
            fname, piece = pieces[index]
            match_template_piece(original_image, piece, restored)
            update_display(restored, vis_label, win_w, win_h)
            vis_window.after(500, process_piece, index + 1)

        process_piece(0)
    else:
        print("랜덤 방식 감지됨 → match_and_paste 방식으로 복원")
        reconstructed, _ = match_and_paste(original_image, pieces, vis_label, update_display)
        cv2.imwrite(os.path.join(restore_dir, "restored.jpg"), reconstructed)
        messagebox.showinfo("완료", "랜덤 방식 이미지 복원이 완료되었습니다.")

def on_close():
    global preview_window, pieces_preview_window, vis_window
    for win in [preview_window, pieces_preview_window, vis_window]:
        if win and win.winfo_exists():
            win.destroy()
    root.destroy()

# UI 구성
root = tk.Tk()
root.title("이미지 조각 복원 프로그램")
root.geometry("300x300")
root.protocol("WM_DELETE_WINDOW", on_close)

font_bold = ("맑은 고딕", 10, "bold")

tk.Button(root, text="원본 이미지 불러오기", font=font_bold, command=load_original_image).pack(pady=10)

split_method = tk.StringVar(value="grid")
tk.Radiobutton(root, text="Grid 방식", font=font_bold, variable=split_method, value="grid").pack()

frame = tk.Frame(root)
frame.pack(pady=5)
tk.Label(frame, text="Rows:", font=font_bold).grid(row=0, column=0)
entry_rows = tk.Entry(frame, width=5)
entry_rows.insert(0, "3")
entry_rows.grid(row=0, column=1)
tk.Label(frame, text="Cols:", font=font_bold).grid(row=0, column=2)
entry_cols = tk.Entry(frame, width=5)
entry_cols.insert(0, "4")
entry_cols.grid(row=0, column=3)

tk.Radiobutton(root, text="랜덤 방식", font=font_bold, variable=split_method, value="random").pack()
tk.Button(root, text="이미지 조각내기", font=font_bold, command=perform_split).pack(pady=10)
tk.Button(root, text="조각 이미지 복원하기", font=font_bold, command=restore_images).pack(pady=10)

root.mainloop()
