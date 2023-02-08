def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                        agnostic=False, multi_label=True, labels=(), max_det=300, merge=False):
    """
    Runs Non-Maximum Suppression (NMS) on inference results
    Params:
         prediction: [batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25]  3个anchor的预测结果总和
         conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
         iou_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
         classes: 是否nms后只保留特定的类别 默认为None
         agnostic: 进行nms是否也去除不同类别之间的框 默认False
         multi_label: 是否是多标签  nc>1  一般是True
         labels:
         max_det: 每张图片的最大目标个数 默认1000
         merge: use merge-NMS 多个bounding box给它们一个权重进行融合  默认False
    Returns:
         [num_obj, x1y1x2y2+object_conf+cls] = [5, 6]
         
    
    这个函数一般会用再detect.py或者test.py的模型前向推理结束之后
    """
    
    # Checks  检查传入的conf_thres和iou_thres两个阈值是否符合范围
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings  设置一些变量
    nc = prediction.shape[2] - 5  # number of classes
    min_wh, max_wh = 2, 4096  # (pixels) 预测物体宽度和高度的大小范围 [min_wh, max_wh]
    max_nms = 30000  # 每个图像最多检测物体的个数  maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # nms执行时间阈值 超过这个时间就退出了 seconds to quit after
    redundant = True  # 是否需要冗余的detections require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # batch_size个output  存放最终筛选后的预测框结果
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    # 定义第二层过滤条件
    xc = prediction[..., 4] > conf_thres  # candidates

    t = time.time()  # 记录当前时刻时间
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # 第一层过滤 虑除超小anchor标和超大anchor   x=[18900, 25]
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height

        # 第二层过滤 根据conf_thres虑除背景目标(obj_conf<conf_thres 0.1的目标 置信度极低的目标)  x=[59, 25]
        x = x[xc[xi]]  # confidence

        # {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        # Cat apriori labels if autolabelling 自动标注label时调用  一般不用
        # 自动标记在非常高的置信阈值（即 0.90 置信度）下效果最佳,而 mAP 计算依赖于非常低的置信阈值（即 0.001）来正确评估 PR 曲线下的区域。
        # 这个自动标注我觉得应该是一个类似RNN里面的Teacher Forcing的训练机制 就是在训练的时候跟着老师(ground truth)走
        # 但是这样又会造成一个问题: 一直靠老师带的孩子是走不远的 这样的模型因为依赖标签数据,在训练过程中,模型会有较好的效果
        # 但是在测试的时候因为不能得到ground truth的支持, 所以如果目前生成的序列在训练过程中有很大不同, 模型就会变得脆弱。
        # 所以个人认为(个人观点): 应该在下面使用的时候有选择的开启这个trick 比如设置一个概率p随机开启 或者在训练的前n个epoch使用 后面再关闭
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # 经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
        if not x.shape[0]:
            continue

        # 计算conf_score
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2) 左上角 右下角   [59, 4]
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # 第三轮过滤:针对每个类别score(obj_conf * cls_conf) > conf_thres    [59, 6] -> [51, 6]
            # 这里一个框是有可能有多个物体的，所以要筛选
            # nonzero: 获得矩阵中的非0(True)数据的下标  a.t(): 将a矩阵拆开
            # i: 下标 [43]   j: 类别index [43] 过滤了两个score太低的
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            # pred = [43, xyxy+score+class] [43, 6]
            # unsqueeze(1): [43] => [43, 1] add batch dimension
            # box[i]: [43,4] xyxy
            # pred[i, j + 5].unsqueeze(1): [43,1] score  对每个i,取第（j+5）个位置的值（第j个class的值cla_conf）
            # j.float().unsqueeze(1): [43,1] class
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)    # 一个类别直接取分数最大类的即可
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class  是否只保留特定的类别  默认None  不执行这里
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 检测数据是否为有限数 Apply finite constraint  这轮可有可无，一般没什么用 所以这里给他注释了
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # 如果经过第三轮过滤该feature map没有目标框了，就结束这轮直接进行下一张图
            continue
        elif n > max_nms:  # 如果经过第三轮过滤该feature map还要很多框(>max_nms)   就需要排序
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # 第4轮过滤 Batched NMS   [51, 6] -> [5, 6]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 做个切片 得到boxes和scores   不同类别的box位置信息加上一个很大的数但又不同的数c
        # 这样作非极大抑制的时候不同类别的框就不会掺和到一块了  这是一个作nms挺巧妙的技巧
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # 返回nms过滤后的bounding box(boxes)的索引（降序排列）
        # i=tensor([18, 19, 32, 25, 27])   nms后只剩下5个预测框了
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights 正比于 iou * scores
            # bounding box合并  其实就是把权重和框相乘再除以权重之和
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]   # 最终输出   [5, 6]

        # 看下时间超没超时  超时没做完的就不做了
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output



