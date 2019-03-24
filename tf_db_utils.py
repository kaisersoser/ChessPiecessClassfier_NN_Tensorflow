import numpy as np
#import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO
from sqlite3 import Error

def create_connection(db_file):   
    """ Create a database connection to the SQLite database
        specified by the db_file
            :param db_file: database file
            :return: Connection object or None    
    """
    
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
        
    return None

def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM PiecesTable")
    
    rows = cur.fetchall()
    
        
def select_all_X_values(conn):
    """
    Query the database and retrieve 
    the values of all images
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT ImageData from PiecesTable")
    
    rows = cur.fetchall()
    
    return rows

def select_pieceID_labels(conn):
    """
    Query the database and return all the PieceIDs 
    """
    cur = conn.cursor()
    cur.execute("SELECT PieceID from PieceLabels")
    
    rows = cur.fetchall()    
    
    return rows


def select_piece_labels(conn, _pieceID):
    """
    Query the database and select a specific row from table matching a piece ID 
    """
    cur = conn.cursor()
    cur.execute("SELECT * from PieceLabels WHERE PieceID=?",(int(_pieceID),))
    
    rows = cur.fetchall()    
    
    return np.array(rows)


def convertToByteIO(imagesArray):
    """
    # Converts an array of images into an array of Bytes
    """
    imagesList = []
    
    for i in range(len(imagesArray)):  
        img = Image.open(BytesIO(imagesArray[i])).convert("RGB")
        imagesList.insert(i, np.array(img))
    
    return imagesList
        
    

def select_all_Y_values(conn):
    """
    Query the database and retrieve the
    values of all labels of images
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT PieceID from PiecesTable")
    
    rows = cur.fetchall()
    
    return rows

def load_dataset(conn):
    """
    Loads the dataset from the database, and returns the X, and Y values for training, 
    validation and test datasets
    :param conn: the connection object
    """
    # Connect to the database and retrieve all X values
    X_dataset = select_all_X_values(conn)
    # Convert the blob list to a numpy byte array
    imagesList = convertToByteIO(np.array(X_dataset))
    full_X_dataset = np.array(imagesList)
    
    print("Size of images dataset:"+str(full_X_dataset.shape))
    
    # Retrieve the labels for the dataset 
    Y_dataset = select_all_Y_values(conn)
    full_Y_dataset = np.array(Y_dataset)
    print("Size of Labels dataset:"+str(full_Y_dataset.shape))
    
    # Use sklearn.train_test_split to split X & Y into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(full_X_dataset, full_Y_dataset, test_size=0.25, random_state=150)
    
    # Reshape the labels array
    y_train = y_train.reshape((1,y_train.shape[0]))
    y_test = y_test.reshape((1,y_test.shape[0]))
    
    # Query and return the list of classes
    classes = np.array(select_pieceID_labels(conn))
    classes = np.squeeze(classes[:])
    
    return X_train, X_test, y_train, y_test, classes
