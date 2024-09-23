import shutil
import json
import os
#read all data from json file

img_links_1 = json.load(open('all_img_file/split.dress.train.json', 'r'))
img_links_2 = json.load(open('all_img_file/split.dress.val.json', 'r'))
img_links_3 = json.load(open('all_img_file/split.dress.test.json', 'r'))

#concatenate all data
src_links = img_links_1 + img_links_2 + img_links_3
for img_path in src_links:
    src_path = os.path.join('resized_images', img_path+".jpg")
    dst_path = os.path.join('dress', img_path+".jpg")
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Warning: Source file not found - {src_path}")
        print("DST PATH: ", dst_path)



# print(new_img_links)

# print(len(new_img_links))

# #move all images to new folder named 'dress'
# import shutil

# # Create the 'dress' folder if it doesn't exist
# dress_folder = 'dress'
# if not os.path.exists(dress_folder):
#     os.makedirs(dress_folder)

# # Move all images to the 'dress' folder
# for img_path in new_img_links:
#     src_path = os.path.join('resized_images', img_path)
#     dst_path = os.path.join(dress_folder, os.path.basename(img_path))
#     if os.path.exists(src_path):
#         shutil.move(src_path, dst_path)
#     else:
#         print(f"Warning: Source file not found - {src_path}")

# print(f"Moved {len(new_img_links)} images to the '{dress_folder}' folder.")


