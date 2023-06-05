'''Charts plotting module
'''

import matplotlib.pyplot as plt
        
        
def plot_loss(history) -> None:
    plt.title(f'Loss\n', size=font_s+4)
    
    history = history.history
    
    plt.plot(history['loss'], '+-r')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.xticks(range(len(history['loss'])))
    plt.grid()
    plt.show()
    
    
def plot_loss_acc(history, valid: bool = True) -> None:    
    plt.title('Accuracy & Loss\n')
    
    history = history.history
    
    loss_acc = ['loss', 'val_loss', 'accuracy', 'val_accuracy'] if valid else ['loss', 'accuracy']
    
    for name in loss_acc:
        label = name if name not in ['loss', 'accuracy'] else f'train_{name}'
        plt.plot(history[name], label=label)
    
    plt.xlabel('epoch')    
    plt.legend(loc='upper right')
    
    plt.xticks(range(len(history['loss'])))
    plt.grid()
    plt.show()