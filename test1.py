import time

t1 = time.time()
def foo(bar, baz):
    time.sleep(2)
    return baz

from multiprocessing.pool import ThreadPool
num_thread = 200
pool = ThreadPool(processes=num_thread)

t2 = time.time()
print(t2 - t1)

result = []
for i in range(num_thread):
    async_result = pool.apply_async(foo, ('world', 'foo')) # tuple of args for foo
    result.append(async_result)
# do some other stuff in the main process

return_val = [res.get() for res in result]  # get the return value from your function.
print(return_val)

t3 = time.time()
print(t3-t2)

def thread_worker(label, score, i):
    ind = label == i
    points = np.array(np.where(ind)).transpose((1, 0))

    if points.shape[0] < 5:
        label[ind] = 0
        return (None, None)

    score_i = np.mean(score[ind])
    if score_i < 0.5:
        label[ind] = 0
        return (None, None)
    
    scale = (1.0, 1.0)
    rect = cv2.minAreaRect(points[:, ::-1])
    bbox = cv2.boxPoints(rect) * scale

    bbox = bbox.astype('int32')
    return (bbox.reshape(-1), score_i)
