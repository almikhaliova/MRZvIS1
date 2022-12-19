from core import *
if __name__ == '__main__':
    print("1) Тренировка\n"
          "2) Компресс\n"
          "3) Декомпресс\n")
    menu = input("Выбор: ")
    if menu == '1':
        filename = 'cat.jpg'
        image = Image.open(filename)
        width, height = image.size
        m = int(input('width: '))
        n = int(input('height: '))
        p = int(input('Кол-во нейронов: '))
        e = int(input('Ошибки: '))
        Train(image, n, m, p, height, width, e, 0.001, 0, [99999])
    elif menu == '2':
        filename = 'cat.jpg'
        compress(filename, [])
    elif menu == '3':
        filename = 'compress_img.bin'
        decompress(filename, (100//10)*3 - 20, [], [], [])
