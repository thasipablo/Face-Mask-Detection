from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # dimensions du cadre
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    # passer le blob dans le reseau et obtenir la detection des images
    faceNet.setInput(blob)
    detections = faceNet.forward()

    print(detections.shape)

    faces = []
    locs = []
    preds = []

    # parcourrir les detections
    for i in range(0, detections.shape[2]):
        # extraire la confidence
        confidence = detections[0, 0, i, 2]

        # filtrer les detections faibles et garder celles des
        # confidence superieurs au minimum
        if confidence > 0.5:
            # calculer les coordonner x et y
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # ajouter les faces a leurs listes correspondantes
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # predire si au-moins une face detecte
    if len(faces) > 0:
        # predictions simultannes au lieu
        # d'une a une pour augmenter la vitesse de prediction
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # retourner les faces et leurs localisations
    return (locs, preds)


# chargement du mask du model de detection des faces
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# chargement du model
maskNet = load_model("mask_detector.model")

# initialiser la video streaming
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # detecter le mask dans la frame video et dire
    # oui ou non il y a port du mask
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # lire les faces detectes
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determiner le classe
        label = "Mask" if mask > withoutMask else "Pas de Mask"
        color = (255, 0, 0) if label == "Mask" else (0, 255, 0)

        # inclure la probabilite dans le label en pourcentage
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        print(label)

        # dessiner le rectangle autour du mask
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # montrer la fenetre
    cv2.imshow("Detection de mask", frame)
    key = cv2.waitKey(1) & 0xFF

    # fermer la fenetre si on appuie sur la touche 'q'
    if key == ord("q"):
        break

# liberer les ressources
cv2.destroyAllWindows()
vs.stop()
