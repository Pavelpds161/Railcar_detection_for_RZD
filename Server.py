'''
Необходимо сначала запустить Server.py. Дождаться, когда в консоли появится сообщение "Запустите Client".
После этого запустить Client.py
Папка с изображениями должна находиться в одной директории с Server.py.
'''
import argparse

from utils.datasets import *
from utils.utils import *
import socket
from PIL import Image
from PIL import ImageOps
import numpy as np
import pickle
from time import sleep
import os
# from detect import detect

# функция для детектирования
def detect(save_img=False):
    out, source, weights, half, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

# Детектирование всех фото в папке images

input_path = "images"
output_path = "output_images"

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights/best_yolov5s_wagons.pt', help='model.pt path')
parser.add_argument('--source', type=str, default=input_path, help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default=output_path, help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()
print(opt)

with torch.no_grad():
    detect()

#===============================================================================================
# Детектирование закончено. Далее всё тот же код по пересылке изображений
#===============================================================================================
input_path = "images"
output_path = "output_images"

def image_to_bytes(img):
    # Функция для сериализации изображения
    byte_array = pickle.dumps(np.array(img))
    return byte_array

# Загружаем все фотографии из папки в список images_list
filenames = os.listdir(output_path)
images_list = []
for filename in filenames:
    #image = Image.open(output_path+'\\'+filename)
    image = Image.open(os.path.join(output_path, filename))
    images_list.append(image_to_bytes(image))

max_bytes_len = max([len(i) for i in images_list])
# Если будет встречаться сериализованный объект, у которого длина значительно меньше, чем у максимального (цифр меньше),
# то заполняем нулями, пока цифр не станет столько же.
zeros_count = 10

sock = socket.socket()
sock.bind(("", 9090))
sock.listen(1)

# Ждём подключения клиента
print("Запустите Client")
conn, addr = sock.accept()

# Пересылаем 12 фотографий
for i in range(12):
    img = images_list[i]
    byte_arr_len = str(len(img)).zfill(zeros_count)
    
    # Пересылаем размер сериализованной фотографии
    conn.send(byte_arr_len.encode("utf8"))
    # Пересылаем саму фотографию
    conn.send(images_list[i])
    
    # Ждём одну секунду
    sleep(1)

# Чтобы выполнение программы резко не прервалось, и клиент успел принять все данные
input("Нажмите Enter для завершения работы программы.")

conn.close()

