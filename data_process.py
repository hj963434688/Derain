import tensorflow as tf
import cv2
import numpy as np
import os
import glob
import time
import numpy as np
import threading
import multiprocessing
import scipy.misc

try:
    import queue
except ImportError:
    import Queue as queue

train_path = '../dataset/derain_dataset/dataset/train'
test_path = '../dataset/derain_dataset/dataset/test'

# image_format = ['jpg', 'jpeg', 'JPG']

image_format = ['png']


class GeneratorEnqueuer():
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)


def get_images(path):
    input_img = []
    label_img = []
    for ext in image_format:
        input_img.extend(glob.glob(
            os.path.join(path + '/input', '*.{}'.format(ext))))
        label_img.extend(glob.glob(
            os.path.join(path + '/label', '*.{}'.format(ext))))

    return input_img, label_img


def generator(input_size=49, batch_size=16):
    image_input_list = np.array(get_images(train_path)[0])
    image_label_list = np.array(get_images(train_path)[1])
    print('{} training images in {}'.format(
        image_input_list.shape[0], train_path))
    index = np.arange(0, image_input_list.shape[0])
    while True:
        np.random.shuffle(index)
        images_input = []
        images_label = []
        for i in index:
            try:
                # im_x = cv2.imread(image_input_list[i])[:, :, ::-1]
                # im_y = cv2.imread(image_label_list[i])[:, :, ::-1]
                im_x = scipy.misc.imread(image_input_list[i], mode='RGB')
                im_y = scipy.misc.imread(image_label_list[i], mode='RGB')
                h, w, _ = im_x.shape
                # print(h, w)
                h_n = h // input_size
                w_n = w // input_size
                # print rd_scale
                # print(h_n, w_n)
                # random crop a area from image
                h_index = np.arange(0, h_n+1)
                w_index = np.arange(0, w_n+1)
                np.random.shuffle(h_index)
                np.random.shuffle(w_index)
                # print(h_index, w_index)
                for j in h_index:
                    for k in w_index:
                        if j < h_n:
                            j1 = j*input_size
                            j2 = (j+1)*input_size
                        else:
                            j1 = h-input_size
                            j2 = h
                        if k < w_n:
                            k1 = k*input_size
                            k2 = (k+1)*input_size
                        else:
                            k1 = w-input_size
                            k2 = w
                        demo_x = im_x[j1:j2, k1:k2, :] / 255
                        demo_y = im_y[j1:j2, k1:k2, :] / 255
                        # print(demo_x.shape)
                        images_input.append(demo_x.astype(np.float32))
                        images_label.append(demo_y.astype(np.float32))
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


def get_test_batch(input_size=48, batch_size=128):
    image_input_list = np.array(get_images(test_path)[0])
    image_label_list = np.array(get_images(test_path)[1])
    # print(image_input_list[0])
    # print(image_label_list[1])
    index = np.arange(0, image_input_list.shape[0])
    while True:
        np.random.shuffle(index)
        images_input = []
        images_label = []
        for i in index:
            try:
                im_x = scipy.misc.imread(image_input_list[i], mode='RGB')
                im_y = scipy.misc.imread(image_label_list[i], mode='RGB')
                h, w, _ = im_x.shape
                # print(h, w)
                h_n = h // input_size
                w_n = w // input_size
                h_index = np.arange(0, h_n + 1)
                w_index = np.arange(0, w_n + 1)
                np.random.shuffle(h_index)
                np.random.shuffle(w_index)
                for j in h_index:
                    for k in w_index:
                        if j < h_n:
                            j1 = j * input_size
                            j2 = (j + 1) * input_size
                        else:
                            j1 = h - input_size
                            j2 = h
                        if k < w_n:
                            k1 = k * input_size
                            k2 = (k + 1) * input_size
                        else:
                            k1 = w - input_size
                            k2 = w
                        demo_x = im_x[j1:j2, k1:k2, :] / 255
                        demo_y = im_y[j1:j2, k1:k2, :] / 255
                        images_input.append(demo_x.astype(np.float32))
                        images_label.append(demo_y.astype(np.float32))
                        if len(images_input) == batch_size:
                            # print('read64-------------')
                            return images_input, images_label

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


if __name__ == '__main__':

    img_generator = get_batch(num_workers=8,
                              input_size=49,
                              batch_size=64)
    data = next(img_generator)
    print(len(data[0]))
    print(data[0][0].shape)
    # print(data[0])
    # img = data[0][0]
    # print(img.shape)
    # cv2.imshow('we', img)
    # cv2.waitKey(0)
    # print(get_images(train_path)[0])
    # print(get_images(train_path)[1])
