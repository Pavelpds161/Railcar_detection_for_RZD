import socket
from PIL import Image, ImageTk
import pickle
import tkinter as tk


# Коннектимся
sock = socket.socket()
sock.connect(('localhost', 9090))

# Принимаем 12 фотографий
images_list = []

for i in range(12):
    
    # Сначала принимаем длину фотографии, которая придёт дальше
    len_data = sock.recv(10)
    
    # Из байт переводим в целочисленное значение
    img_len = int(len_data.decode("utf8"))

    # Принимаем фото
    img_data = sock.recv(img_len)
    
    # Добавляем фото в список
    images_list.append(Image.fromarray(pickle.loads(img_data)))

# Сохраняем картинки в jpg
for i, img in enumerate(images_list):
    img.save(str(i) + ".jpg")

# tkinter для вывода на экран
roots = []

for i in range(12):
    roots.append(tk.Tk())
    img = ImageTk.PhotoImage(file=str(i)+".jpg")
    panel = tk.Label(roots[-1], image = img)
    panel.pack()

    roots[-1].mainloop()

sock.close()

