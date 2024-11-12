import numpy as np
import matplotlib.pyplot as plt

def display_image(x_test, predictions):
    fig, ax = plt.subplots()
    current_idx = 0
    
    def _display_image(idx):
        ax.clear() # Clear previous plot
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Predicted label: {np.argmax(predictions[idx])}")
        plt.draw()
        
    def on_key(event):
        nonlocal current_idx
        if event.key == 'right': # Right arrow key to go next
            current_idx += 1
            if current_idx >= len(x_test):
                print('End of test set...')
                plt.close()
            else:
                _display_image(current_idx)
        elif event.key == 'escape': # Escape key to exit
            plt.close()
            
    # Key event listener
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    _display_image(current_idx)
    
    plt.show()