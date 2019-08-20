import os
import cv2
import random
import shutil

input_path = './dataset/input/'
label_path = './dataset/label/'

train_input_path = './dataset/train/input/'
train_label_path = './dataset/train/label/'
test_input_path = './dataset/test/input/'
test_label_path = './dataset/test/label/'


def makdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def trun():
    path = "./dataset/train/label/"
    print(path)

    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.png':
            # print(filename)
            img = cv2.imread(path + filename)
            print(filename.replace(".png", ".jpg"))
            newfilename = filename.replace(".png", ".jpg")
            # cv2.imshow("Image",img)
            # cv2.waitKey(0)

            cv2.imwrite(path + newfilename, img)


def test():
    img1 = cv2.imread('./dataset/train/input/1.jpg')
    img2 = cv2.imread('./dataset/train/input/1.png')
    print()
    print(img1[0][0])
    print(img2[0][0])
    # print(img1 == img2)


def split_dataset(ratio = 0.2):
    makdir(train_input_path)
    makdir(train_label_path)
    makdir(test_input_path)
    makdir(test_label_path)

    in_files = [filename for filename in os.listdir(input_path)]
    la_files = [filename for filename in os.listdir(label_path)]
    print(in_files)
    if in_files != la_files:
        print('数据不一致请检查文件')
    else:
        random.shuffle(in_files)
        print(in_files)
    boundary = int(len(in_files) * ratio)
    file_test = in_files[:boundary]
    file_train = in_files[boundary:]
    print('训练文件:{}, 测试文件:{}'.format(len(file_train), len(file_test)))
    print(file_test)
    print(file_train)

    for f in file_test:
        shutil.copyfile(os.path.join(input_path, f), os.path.join(test_input_path, f))
        shutil.copyfile(os.path.join(label_path, f), os.path.join(test_label_path, f))
    for f in file_train:
        shutil.copyfile(os.path.join(input_path, f), os.path.join(train_input_path, f))
        shutil.copyfile(os.path.join(label_path, f), os.path.join(train_label_path, f))


if __name__ == '__main__':
    test()
