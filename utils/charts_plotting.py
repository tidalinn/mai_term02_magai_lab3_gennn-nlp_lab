'''Charts plotting module
'''

import matplotlib.pyplot as plt
        

def plot_loss_acc(history, valid: bool = False) -> None:   
    plt.figure(figsize=(6,5))
    plt.title('Accuracy & Loss\n')
    
    history = history.history
    
    loss_acc = ['loss', 'val_loss', 'accuracy', 'val_accuracy'] if valid else ['loss', 'accuracy']
    
    for name in loss_acc:
        label = name if name not in ['loss', 'accuracy'] else f'train_{name}'
        plt.plot(history[name], label=label)
    
    plt.xlabel('epoch')    
    plt.legend(loc='upper right')
    
    plt.grid()
    plt.show()