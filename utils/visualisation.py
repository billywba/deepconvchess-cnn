import matplotlib.pyplot as plt


def display_cropped_chess_board(board, labels = None):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(wspace=0.75)

    for i in range(0, 64):
        fig.add_subplot(8, 8, i + 1)
        
        plt.imshow(board[i])
        plt.title(labels[i])
        
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
