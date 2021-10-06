from multiprocessing.pool import ThreadPool

def thread_worker(list_points, score, scale, min_area, min_score, i):
    points = np.array(list_points[i-1]).transpose((1, 0))
    ind = list_point[i-1]

    if points.shape[0] < min_area:
        label[ind] = 0
        return (None, None)

    score_i = np.mean(score[ind])
    if score_i < min_score:
        label[ind] = 0
        return (None, None)
    
    rect = cv2.minAreaRect(points[:, ::-1])
    bbox = cv2.boxPoints(rect) * scale

    bbox = bbox.astype('int32')
    return (bbox.reshape(-1), score_i)

####### thay cho vong for 

bboxes = []
scores = []

# num_thread = label_num -1 
pool = ThreadPool(processes=label_num-1)

result = []
for i in range(1, label_num):
    async_result = pool.apply_async(thread_worker, (label, score, scale, cfg.test_cfg.min_area, cfg.test_cfg.min_score, i)) # tuple of args for foo
    result.append(async_result)

for res in result:
    box = res.get()[0]
    if np.any(box != None):
        bboxes.append(box)
        scores.append(res.get()[1])

print('num word', print(len(bboxes)))