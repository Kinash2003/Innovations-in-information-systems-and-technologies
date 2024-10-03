from deepface import DeepFace

# verification = DeepFace.verify(img1_path="img1.jpg", img2_path="img2.jpg")
# for key, value in verification.items():
#     if isinstance(value, dict):
#         print(f'"{key}":{{')
#         for sub_key, sub_value in value.items():
#             print(f'     \"{sub_key}\": {sub_value}')
#         print(f'}}')
#     else:
#         print(f'"{key}": {value},')

#recognition = DeepFace.find(img_path = "img1.jpg", db_path = "D:/Lab3/facial")
#print(f'{recognition}')

# analysis = DeepFace.analyze(img_path="img1.jpg", actions=["age", "gender", "emotion", "race"])
# for result in analysis:
#     for result in analysis:
#         for key, value in result.items():
#             if isinstance(value, dict):
#                 print(f'"{key}":{{')
#                 for sub_key, sub_value in value.items():
#                     print(f'     \"{sub_key}\": {sub_value}')
#                 print(f'}}')
#             else:
#                 print(f'"{key}": {value},')
# DeepFace.stream(db_path = "D:/Lab3/facial")

# models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
# #face verification
# verification = DeepFace.verify("img1.jpg", "img2.jpg", model_name = models[1])
# #face recognition
# recognition = DeepFace.find(img_path = "img1.jpg", db_path = "D:/Lab3/facial", model_name = models[1])
# print(f'{recognition}')


detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"] #face verification
verification = DeepFace.verify("img1.jpg", "img2.jpg", detector_backend = detectors[1])
#face recognition
recognition = DeepFace.find(img_path = "img1.jpg", db_path = "D:/Lab3/facial", detector_backend = detectors[1])
print(f'{recognition}')
# detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
# img1 = DeepFace.detectFace("img1.jpg", detector_backend = detectors[0])
# print(f'{img1}')