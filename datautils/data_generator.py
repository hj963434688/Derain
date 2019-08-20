import time
import numpy as np
import scipy.misc
from datautils.data_queue import GeneratorEnqueuer
from datautils import *


def generator(data_path='../dataset/derain/my/train/', input_size=88, batch_size=128):
    input_list = np.array(get_data_files(data_path)[0])
    label_list = np.array(get_data_files(data_path)[1])
    total_num = len(input_list)
    print('{} training images in {}'.format(
        total_num, data_path))
    n = 4
    index = np.arange(0, total_num)
    group_num = total_num // n
    images_input = []
    images_label = []
    while True:
        np.random.shuffle(index)
        for i in range(group_num):
            try:
                group_index = index[n * i:n*(i+1)]
                group_images = []
                group_labels = []
                group_shapes = []
                for ix in group_index:
                    group_images.append(scipy.misc.imread(input_list[ix], mode='RGB'))
                    group_labels.append(scipy.misc.imread(label_list[ix], mode='RGB'))
                    group_shapes.append(group_images[-1].shape)
                # print(im_x.shape)
                # cv2.imshow('im', group_images[0])
                # cv2.waitKey(0)
                # h, w, _ = im_x.shape
                # print(h, w)
                # h_n = h // input_size
                # w_n = w // input_size
                # h_index = np.arange(0, h_n + 1)
                # w_index = np.arange(0, w_n + 1)
                i_index = np.arange(0, n)
                h_index = np.arange(0, 100)
                w_index = np.arange(0, 100)
                np.random.shuffle(i_index)
                np.random.shuffle(h_index)
                np.random.shuffle(w_index)

                for j in h_index:
                    for k in w_index:
                        for m in i_index:
                            h, w, _ = group_shapes[m]
                            h_ix = h // 100
                            w_ix = w // 100
                            if j * h_ix + input_size < h:
                                j1 = j * h_ix
                                j2 = j1 + input_size
                            else:
                                j1 = h - input_size
                                j2 = h
                            if k * w_ix + input_size < w:
                                k1 = k * w_ix
                                k2 = k1 + input_size
                            else:
                                k1 = w - input_size
                                k2 = w
                            demo_x = group_images[m][j1:j2, k1:k2, :] / 255
                            demo_y = group_labels[m][j1:j2, k1:k2, :] / 255
                            images_input.append(demo_x.astype(np.float32))
                            images_label.append(demo_y.astype(np.float32))
                            # cv2.imshow('xx', group_images[m])
                            # cv2.imshow('x', demo_x)
                            # cv2.waitKey(0)
                            if len(images_input) == batch_size:
                                # print('read64-------------')
                                yield images_input, images_label
                                images_input = []
                                images_label = []


            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    img_generator = get_batch(num_workers=1,
                              data_path=work_place + 'dataset/derain/my/train/',
                              input_size=88,
                              batch_size=64)
    for i in range(100):
        data = next(img_generator)
        print(i)
        print(data[0][0].shape)
