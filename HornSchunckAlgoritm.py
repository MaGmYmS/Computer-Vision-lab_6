import cv2
import numpy as np


def horn_schunck(video_path, lambda_val=0.01, threshold=0.01):
    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)

    # Первый кадр
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = np.float32(prev_frame_gray)

    while True:
        # След кадр
        ret, next_frame = cap.read()
        if not ret:
            break

        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_frame_gray = np.float32(next_frame_gray)

        # Градиенты
        Ix = cv2.Sobel(prev_frame_gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(prev_frame_gray, cv2.CV_64F, 0, 1, ksize=5)
        It = next_frame_gray - prev_frame_gray

        # Поле опт потоков u - x v -y
        u = np.zeros_like(prev_frame_gray)
        v = np.zeros_like(prev_frame_gray)

        # Итерации решения уравнений Хорна-Шанка
        for _ in range(5):
            u_avg = cv2.blur(u, (5, 5))
            v_avg = cv2.blur(v, (5, 5))

            numerator = Ix * u_avg + Iy * v_avg + It
            denominator = lambda_val ** 2 + Ix ** 2 + Iy ** 2

            # Обновление значений u и v
            u = u_avg - Ix * numerator / denominator
            v = v_avg - Iy * numerator / denominator

        # Вычисление величины векторов оптического потока
        magnitude = np.sqrt(u ** 2 + v ** 2)
        # Создание маски для значительных движений
        motion_mask = magnitude > threshold

        # отображение
        motion_image = np.zeros_like(next_frame)
        motion_image[motion_mask] = [0, 0, 255]  # Красный цвет для выделения движения

        # Отображение изображения с движением
        cv2.imshow('Motion', motion_image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Обновление предыдущего кадра для следующей итерации
        prev_frame_gray = next_frame_gray.copy()

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


# Пример использования функции
lambda_value = 100
T_value = 0.1
# video_path = 'Barber pole.mp4'
video_path = 'video.mp4'
horn_schunck(video_path, lambda_val=lambda_value, threshold=T_value)
