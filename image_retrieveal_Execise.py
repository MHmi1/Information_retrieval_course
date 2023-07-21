
#this is th exercise of information retrieval course - design a image retrieval system
#I have used VGG16 pretrained Deep Neural network and clustering together to reach suitable results
#all images in oxford building dataset ars 768 * 1024 px and this code based on this dataset
import os
import numpy as np
import sqlite3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


#Extracts features from an image using the VGG16 model.
def extract_vgg16_features(img_path, target_size=(224, 224)):
    image = load_img(img_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]

    model = VGG16(include_top=False, weights='imagenet', input_shape=(target_size[0], target_size[1], 3))
    features = model.predict(image)
    features = features.reshape((-1,))

########################################################################

#Builds an index of image features using k-means clustering
def build_index(img_paths, num_clusters=50, target_size=(224, 224)):
    all_features = []
    for img_path in img_paths:                  #add all unique images in path to build index
        features = extract_vgg16_features(img_path, target_size)
        if features is not None:
            all_features.append(features)

    all_features = np.array(all_features)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_features)
    return kmeans, all_features   # Return Array of image features used for clustering

########################################################################


# Saves image features to an SQLite DB.
def save_features_to_DB(DB_path, img_paths, all_features):
    conn = sqlite3.connect(DB_path)       #establish and intilize connection
    cursor = conn.cursor()  

    cursor.execute('DROP TABLE IF EXISTS features')
     #we have two columns in the DB : first is img_address(TEXT) and seconde is vector of image features(BLOB)
    cursor.execute('''CREATE TABLE features (img_path TEXT PRIMARY KEY, features BLOB)''') 

    for i, img_path in enumerate(img_paths):
        features_blob = all_features[i].tobytes()
        cursor.execute('INSERT INTO features (img_path, features) VALUES (?, ?)', (img_path, features_blob))
    
    conn.commit()
    conn.close()
    
########################################################################

#    Loads image features from an SQLite DB.
def load_features_from_DB(DB_path):
    conn = sqlite3.connect(DB_path)
    cursor = conn.cursor()

    cursor.execute('SELECT img_path, features FROM features')
    rows = cursor.fetchall()

    img_paths = []
    all_features = []

    for row in rows:
        img_paths.append(row[0])
        features_blob = row[1]
        features = np.frombuffer(features_blob, dtype=np.float32)
        all_features.append(features)  #append all features of the images to all_features = [] 

    conn.close()
    return img_paths, np.array(all_features)  #return all path and all features

########################################################################


    #Calculates the cosine similarity between two feature vectors
def cal_cosine_similarity(features1, features2):

    return cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

########################################################################


#    Retrieves similar images based on a query image.
def retrieve_similar_imgs(query_img_path, kmeans, all_features, num_results=3, target_size=(224, 224)):    
    """
    tasks are Retrieves similar images based on a query image.

    input Parameters:
        query_image_path (str): The path of the query image.
        kmeans (KMeans): The trained k-means clustering model.
        all_features (numpy array): Array of image features used for clustering.
        num_results (int, optional): Number of similar images to retrieve.
        target_size (tuple, optional): The target size for resizing the image before feature extraction.
                                    
    Returns:
        sorted_images (list): List of file paths of similar images, sorted by similarity.
    """
    query_features = extract_vgg16_features(query_img_path, target_size)
    query_features = query_features.reshape(1, -1)  # Reshape to 2D array

    query_cluster = kmeans.predict(query_features)    # Predict the cluster of the query image using the trained k-means model
    similar_img_indices = np.where(kmeans.labels_ == query_cluster)  #     # Find indices of images in the same cluster as the query image

    # Initialize lists to store similar images and their creative similarities
    similar_imgs = []
    my_similarities = []
    
    for idx in similar_img_indices[0]:
        img_path = img_paths[idx]
        img_features = all_features[idx]
        img_features = img_features.reshape(1, -1)  # Reshape to 2D array
        similarity = cal_cosine_similarity(query_features, img_features)   # Calculate the cosine similarity between the query image and the current image
        similar_imgs.append(img_path)          # Append the image path and similarity to the corresponding lists
        my_similarities.append(similarity)

    sorted_indices = np.argsort(my_similarities)[::-1]     # Sort the images based on the cosine similarity in descending order
    sorted_imgs = [similar_imgs[idx] for idx in sorted_indices]
    return sorted_imgs[:num_results]        # Return List of file paths of similar images, sorted by similarity.


if __name__ == "__main__":
    dataset_path = r'C:\Users\Lion\Desktop\python\oxbuild_images'
    query_image_path = r'C:\Users\Lion\Desktop\python\queries\radcliffe_camera_000410.jpg'
    database_path = r'C:\Users\Lion\Desktop\python\db.db'

    img_paths = []
    for file in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, file)
        img_paths.append(img_path)

    print("Number of imgs in dataset:", len(img_paths))
    kmeans, all_features = build_index(img_paths, num_clusters=len(img_paths), target_size=(224, 224))
    save_features_to_DB(DB_path, img_paths, all_features)

    # After saving features to the DB, you can load them directly from the DB for fast retrieval
    loaded_img_paths, loaded_all_features = load_features_from_DB(DB_path)

    num_results = 3
    similar_imgs = retrieve_similar_imgs(query_img_path, kmeans, loaded_all_features, num_results, target_size=(224, 224))

    # Calculate cosine similarity for each pair of imgs
    query_features = extract_vgg16_features(query_img_path, target_size=(224, 224))
    my_similarities = []  #this similarity based on the cosine similarity metric
    for img_path in similar_imgs:
        img_features = extract_vgg16_features(img_path, target_size=(224, 224))
        similarity = cal_cosine_similarity(query_features, img_features)
        my_similarities.append(similarity)

    # Display or process the retrieved similar imgs and creative similarities as needed
    print("-> Similar imgs and cosine similarities:")
    for img_path, similarity in zip(similar_imgs, my_similarities):
        print(f"img: {img_path}, cosine Similarity: {similarity:.4f}")