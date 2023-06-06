import numpy as np
import mediapipe as mp
import cv2 as cv
import os
import glob

def Iris_Segmentation(image):
    img = image.copy()
    mp_face_mesh = mp.solutions.face_mesh

    LEFT_IRIS = [474,475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        results = face_mesh.process(rgb_frame)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
            for p in results.multi_face_landmarks[0].landmark])
            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(image, center_left, int(l_radius), (0,255,0), 1, cv.LINE_AA)
            cv.circle(image, center_right, int(r_radius), (0,255,0), 1, cv.LINE_AA)

            cv.circle(mask, center_left, int(l_radius), (255,255,255), -1, cv.LINE_AA)
            cv.circle(mask, center_right, int(r_radius), (255,255,255), -1, cv.LINE_AA)

    return mask, image

def load_images(path):
    image_list = []
    for filename in glob.glob(path):
        im=cv.imread(filename)
        #im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        image_list.append(im)

    return image_list

images = load_images("Dataset_Faces/*.jpg")

mask_path_training = 'Dataset_Faces_Mask_training' 
if not os.path.exists(mask_path_training):
    os.makedirs(mask_path_training)

mask_path_validation = 'Dataset_Faces_Mask_validation' 
if not os.path.exists(mask_path_validation):
    os.makedirs(mask_path_validation)

segmentation_path_training = 'Dataset_Faces_training' 
if not os.path.exists(segmentation_path_training):
    os.makedirs(segmentation_path_training)

segmentation_path_validation = 'Dataset_Faces_validation' 
if not os.path.exists(segmentation_path_validation):
    os.makedirs(segmentation_path_validation)

c = 1

for image in images:
    img = image.copy()
    segmentation = Iris_Segmentation(img)

    if c <= 800:
        cv.imwrite(mask_path_training + "/"+ str(c)+ ".jpg", segmentation[0])
        #cv.imwrite(segmentation_path_training + "/"+ str(c)+ ".jpg", segmentation[1])
        cv.imwrite(segmentation_path_training + "/"+ str(c)+ ".jpg", image)
    else:
        cv.imwrite(mask_path_validation + "/"+ str(c)+ ".jpg", segmentation[0])
        #cv.imwrite(segmentation_path_validation + "/"+ str(c)+ ".jpg", segmentation[1])
        cv.imwrite(segmentation_path_validation + "/"+ str(c)+ ".jpg", image)

    c += 1
