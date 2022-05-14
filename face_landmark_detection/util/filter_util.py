import cv2
import csv
import numpy as np

from . import face_blend_common as fbc


FILTER_CONFIGS = {
    'anonymous':
        [{'path': "face_landmark_detection/filters/anonymous.png",
          'anno_path': "face_landmark_detection/filters/anonymous_annotations.csv",
          'morph': True,
          'has_alpha': True}],
    'anime':
        [{'path': "face_landmark_detection/filters/anime.png",
          'anno_path': "face_landmark_detection/filters/anime_annotations.csv",
          'morph': True,
          'has_alpha': True}],
    'dog':
        [{'path': "face_landmark_detection/filters/dog-ears.png",
          'anno_path': "face_landmark_detection/filters/dog-ears_annotations.csv",
          'morph': False,
          'has_alpha': True},
         {'path': "face_landmark_detection/filters/dog-nose.png",
          'anno_path': "face_landmark_detection/filters/dog-nose_annotations.csv",
          'morph': False,
          'has_alpha': True}],
    'cat':
        [{'path': "face_landmark_detection/filters/cat-ears.png",
          'anno_path': "face_landmark_detection/filters/cat-ears_annotations.csv",
          'morph': False,
          'has_alpha': True},
         {'path': "face_landmark_detection/filters/cat-nose.png",
          'anno_path': "face_landmark_detection/filters/cat-nose_annotations.csv",
          'morph': False,
          'has_alpha': True}],
    'jason-joker':
        [{'path': "face_landmark_detection/filters/jason-joker.png",
          'anno_path': "face_landmark_detection/filters/jason-joker_annotations.csv",
          'morph': True,
          'has_alpha': True}],
    'gold-crown':
        [{'path': "face_landmark_detection/filters/gold-crown.png",
          'anno_path': "face_landmark_detection/filters/gold-crown_annotations.csv",
          'morph': False,
          'has_alpha': True}],
    'flower-crown':
        [{'path': "face_landmark_detection/filters/flower-crown.png",
          'anno_path': "face_landmark_detection/filters/flower-crown_annotations.csv",
          'morph': False,
          'has_alpha': True}],
}


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))

    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # RGBA => BGR, A
    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


def load_landmarks(annotation_path):
    with open(annotation_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # Get landmark
        points = {}
        for i, row in enumerate(csv_reader):
            # row : Index, X, Y, ...
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
  
        return points


def load_filter(filter_name="dog"):
    # Load filter info
    filters = FILTER_CONFIGS[filter_name]

    # Iterate over sub-filter
    multi_filter_runtime = []

    for filter in filters:
        # Initialize sub-filter info
        temp_dict = {}

        # Load filter image
        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        # Load filter landmarks
        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        # Get filter's delaunay triangles
        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime
