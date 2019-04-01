import tensorflow as tf
import numpy as np
import threading
import time


def MyLoop(coord, worker_id):
    while not coord.should_stop():
        # stop all threads randomly
        if np.random.rand() < 0.1:
            print('stopping from id: %d' %(worker_id))
            coord.request_stop()
        else:
            print('working on id: %d' %(worker_id))
        time.sleep(1)


def multi_threads_test():
    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=MyLoop, args=(coord,i,)) for i in range(5)]
    for t in threads:
        t.start()
    coord.join(threads)


def queue_op_test():
    queue = tf.FIFOQueue(100, "float")
    enqueue_op = queue.enqueue([tf.random_normal([1])])
    dequeue_op = queue.dequeue()

    # starts 5 threads with enqueue_op operation on queue
    queue_runner = tf.train.QueueRunner(queue, [enqueue_op] * 5)
    tf.train.add_queue_runner(queue_runner)

    with tf.Session() as sess:
        # start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # print 3 values in queue
        for _ in range(3):
            print(sess.run(dequeue_op)[0])
        
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # multi_threads_test()
    queue_op_test()
