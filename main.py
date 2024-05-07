import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve as filter2


def draw_quiver(u, v, before_img):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(before_img, cmap='gray')

    magnitude_avg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx ** 2 + dy ** 2) ** 0.5
            # draw only significant changes
            if magnitude > magnitude_avg:
                ax.arrow(j, i, dx, dy, color='red')

    plt.draw()
    plt.show()


def get_magnitude(u, v):
    scale = 3
    sum_magnitude = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            counter += 1
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx ** 2 + dy ** 2) ** 0.5
            sum_magnitude += magnitude

    mag_avg = sum_magnitude / counter

    return mag_avg


def get_derivatives(img1, img2):
    # derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1, x_kernel) + filter2(img2, x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx, fy, ft]


def computeHS(before_img, after_img, alpha, delta):
    # removing noise
    before_img = cv2.GaussianBlur(before_img, (5, 5), 0)
    after_img = cv2.GaussianBlur(after_img, (5, 5), 0)

    # set up initial values
    u = np.zeros((before_img.shape[0], before_img.shape[1]))
    v = np.zeros((before_img.shape[0], before_img.shape[1]))
    fx, fy, ft = get_derivatives(before_img, after_img)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                           [1 / 6, 0, 1 / 6],
                           [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha ** 2 + fx ** 2 + fy ** 2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        # converges check (at most 300 iterations)
        if diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    draw_quiver(u, v, before_img)

    return [u, v]


def extract_frames(video_path_inner):
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path_inner)

    # Проверяем, открыт ли файл
    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame1 = None
    # Читаем видеофайл кадр за кадром
    for i in range(frame_count):
        ret, frame = cap.read()

        # Проверяем, успешно ли был прочитан кадр
        if not ret:
            print(f"Ошибка при чтении кадра {i}")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame2 = frame1
        frame1 = gray_frame
        if i != 0:
            u, v = computeHS(frame1, frame2, alpha=15, delta=10 ** -1)

    # Закрываем видеофайл после обработки
    cap.release()
    print("Обработано кадров:", frame_count)


if __name__ == '__main__':
    video_path = "videoplayback.mp4"
    extract_frames(video_path)
