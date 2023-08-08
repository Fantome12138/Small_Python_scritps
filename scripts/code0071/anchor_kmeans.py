import numpy as np


def iou(boxes, anchors):
    """
    Calculate the IOU between boxes and anchors.

    :param boxes: 2-d array, shape(n, 2)
    :param anchors: 2-d array, shape(k, 2)
    :return: 2-d array, shape(n, k)
    """
    # Calculate the intersection,
    # the new dimension are added to construct shape (n, 1) and shape (1, k),
    # so we can get (n, k) shape result by numpy broadcast
    w_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
    h_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
    inter = w_min * h_min
       
    # Calculate the union
    box_area = boxes[:, 0] * boxes[:, 1]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union = box_area[:, np.newaxis] + anchor_area[np.newaxis]

    return inter / (union - inter)

def fit(self, boxes):
        """
        Run K-means cluster on input boxes.

        :param boxes: 2-d array, shape(n, 2), form as (w, h)
        :return: None
        """
        # If the current number of iterations is greater than 0, then reset
        if self.n_iter > 0:
            self.n_iter = 0

        np.random.seed(self.random_seed)
        n = boxes.shape[0]

        # Initialize K cluster centers (i.e., K anchors)
        self.anchors_ = boxes[np.random.choice(n, self.k, replace=True)]

        self.labels_ = np.zeros((n,))

        while True:
            self.n_iter += 1

            # If the current number of iterations is greater than max number of iterations , then break
            if self.n_iter > self.max_iter:
                break

            self.ious_ = self.iou(boxes, self.anchors_)
            distances = 1 - self.ious_
            cur_labels = np.argmin(distances, axis=1)

            # If anchors not change any more, then break
            if (cur_labels == self.labels_).all():
                break

            # Update K anchors
            for i in range(self.k):
                self.anchors_[i] = np.mean(boxes[cur_labels == i], axis=0)

            self.labels_ = cur_labels