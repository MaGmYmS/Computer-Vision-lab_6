import cv2
import numpy as np


def goodFeaturesToTrack(image, draw_corners=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Поиск особенных точек с использованием функции goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Преобразуем координаты точек в целочисленные значения
    corners_return = np.float32(corners)

    if draw_corners:
        corners_draw = np.int32(corners)
        # Рисуем круги на найденных точках
        for corner in corners_draw:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # Отображаем изображение с найденными точками
        cv2.imshow("Corners", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return corners_return


def visualize_optical_flow(point_new, point_old, state, frame):
    # Отрисовка стрелок на изображении
    good_new = point_new[state == 1]
    good_old = point_old[state == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if np.sqrt((a - c) ** 2 + (b - d) ** 2) > 0.5:
            frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

    return frame.astype(np.uint8)


def video_stream_processing(video_path_inner, method, **kwargs):
    cap = cv2.VideoCapture(video_path_inner)

    # Читаем первый кадр
    ret, frame1 = cap.read()

    while True:
        # Читаем следующий кадр
        ret, frame2 = cap.read()

        if not ret:
            break

        # Вычисляем оптический поток
        (new_points, state, error), old_points = method(frame1=frame1, frame2=frame2, kwargs=kwargs)

        # Визуализируем оптический поток
        frame_with_flow = visualize_optical_flow(new_points, old_points, state, frame1)

        # Отображаем кадр с оптическим потоком
        cv2.imshow('Optical Flow', frame_with_flow)

        # Переходим к следующему кадру
        frame1 = frame2

        # Для остановки видео по нажатию клавиши 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Закрываем видеофайл после обработки
    cap.release()
    cv2.destroyAllWindows()


def methodLK(frame1, frame2, kwargs):
    prev_pts = goodFeaturesToTrack(frame1, False)
    next_pts = goodFeaturesToTrack(frame2, False)
    if "k" in kwargs.keys():
        k = int(kwargs["k"])
        flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, prev_pts, next_pts, winSize=(k, k))
    else:
        flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, prev_pts, next_pts)

    return flow, prev_pts


# Пример использования метода
video_path = "videoplayback.mp4"
video_stream_processing(video_path, methodLK)
