import os
import random
import shutil
import pathlib

import numpy as np

from sklearn.model_selection import train_test_split

from PIL import Image

import chess


def populate_test_folder(dataset_dir_path):
    os.makedirs(dataset_dir_path / "test", exist_ok=True)

    real_dataset_images_list = os.listdir(dataset_dir_path / "real")
    random.shuffle(real_dataset_images_list)

    files_to_move = real_dataset_images_list[0:int(len(real_dataset_images_list) * 0.25)]
    for file_name in files_to_move:
        shutil.move(dataset_dir_path / "real" / file_name, dataset_dir_path / "test" / file_name)


def load_dataset(dataset_path):
    X = []
    y = []
    
    for file in dataset_path.iterdir():
        if file.is_file() and file.name.endswith((".jpg", ".jpeg")):
            print("Processing %s" % dataset_path.name + "/" + file.name)

            # Load FEN into Chess board to get the label later on
            fen = file.name.split(".")[0] # Remove file extension
            fen = fen.replace("_", "/")
            board = chess.Board(fen)
            
            # Open image of Chess board
            img = Image.open(dataset_path / file.name)
            IMG_WIDTH, IMG_HEIGHT = img.size

            # Crop each image into 64 sections, one for each tile on the board
            for i in range(8):
                for j in range(8):
                    crop_x = j * IMG_WIDTH // 8
                    crop_y = i * IMG_HEIGHT // 8
                    img_crop = img.crop((crop_x, crop_y, crop_x + IMG_WIDTH // 8, crop_y + IMG_HEIGHT // 8))

                    piece = board.piece_at(((7 - i) * 8) + j)
                    if piece is not None:
                        piece = piece.symbol()
                    else:
                        piece = '_'

                    X.append(np.array(img_crop))
                    y.append(piece)

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_train_valid_test_dataset(dataset_dir_path):
    X_rendered, y_rendered = load_dataset(dataset_dir_path / "render")
    X_real, y_real = load_dataset(dataset_dir_path / "real")
    X_test, y_test = load_dataset(dataset_dir_path / "test")

    X = np.concatenate((X_rendered, X_real))
    y = np.concatenate((y_rendered, y_real))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    return X_train, y_train, X_val, y_val, X_test, y_test