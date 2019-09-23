import math, gc, os, sys
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


class LRFinder(object):
    """
    Plots the change of the loss function of a TensorFlow Keras model when the learning rate is exponentially increasing.
    
    Usage:
        ```python
        lrfind = LRFinder(model)
        lrfind.find((x_train, y_train), 25000, 32, 1e-7, 10, steps=500)
        lrfind.plot()
        ```    
    References:
        1. https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
        2. https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, model, **kwargs):
        self.model = model
        
    def find(self, dataset, n_obs:int, bs:int=32, 
                start_lr:float=1e-7, end_lr:float=10, 
                steps:int=200, linear:bool=False, stop_divergence:bool=True):
        """Runs learning rate search procedure.
        
        Args:
            dataset:                TensorFlow Dataset(tf.data.Dataset) or tuple (X, y)
            n_obs (int):            sample size of the dataset
            bs (int):               batch size
            start_lr (float):       initial learing rate
            end_lr (float):         end learning rate
            steps (int):            number of batch updates to perform the search
            linear (bool):          linear or exponential multiplier to the learning rate
            stop_divergence (bool): check for loss explosion and stop further training
            
        """
        ratio = end_lr/float(start_lr)
        steps_per_epoch = n_obs/bs
        lr_mult = (ratio/steps) if linear else ratio**(1/steps)
        epochs = int(np.ceil(steps/steps_per_epoch))
        cb = LRFinderCallback(start_lr, lr_mult, steps, linear, stop_divergence)
        if isinstance(dataset, tf.data.Dataset):
            self.model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[cb])
        else:
            X_train, y_train = dataset
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=bs, callbacks=[cb])
        self.cb = cb
        
    def get_derivatives(self, sma):
        if sma < 1:
            raise ValueError('Arg `sma`: must be integer and greater than or equal to 1')
        derivatives = [0] * sma
        lrs = self.cb.lrs
        losses = self.cb.losses
        for i in range(sma, len(lrs)):
            derivatives.append((losses[i] - losses[i - sma]) / sma)
        min_lr = lrs[np.argmin(derivatives)]
        return derivatives, min_lr
        
    def _split_list(self, vals, skip_start, skip_end):
        return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]
        
    def plot_lr(self, skip_start:int=10, skip_end:int=5):
        """Plots learning rate over iterations.
        
        Args:
            skip_start (int): lr's to skip from beginning
            skip_end (int): lr's to skip from end
        """
        lrs = self._split_list(self.cb.lrs, skip_start, skip_end)
        iterations = self._split_list(list(range(len(self.cb.lrs))), skip_start, skip_end)
        fig, ax = plt.subplots(1,1)
        ax.plot(iterations, lrs)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Learning Rate')
        
    def plot_loss(self, skip_start:int=10, skip_end:int=1, smooth:bool=False, suggestion:bool=True):
        """Plots learning rates vs. loss."""
        lrs = self._split_list(self.cb.lrs, skip_start, skip_end)
        losses = self._split_list(self.cb.losses, skip_start, skip_end)
        if smooth:
            xs = np.arange(len(losses))
            spl = UnivariateSpline(xs, losses)
            losses = spl(xs)
        fig, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        if suggestion:
            #try: mg = (np.gradient(np.array(losses))).argmin()
            #except:
            #    print("Failed to compute the gradients, there might not be enough points.")
            #    return
            #print(f"Min numerical gradient: {lrs[mg]}")
            #ax.plot(lrs[mg],losses[mg],markersize=5,marker='o',color='red')
            #self.min_grad_lr = lrs[mg]
            print('Suggestions:')
            ml = np.argmin(losses)
            print(f'Learning rate: {lrs[ml]}, Minimum Loss: {np.min(losses)}')
            print(f"Learning rate at min. loss divided by 10: {lrs[ml]/10}")
            
    def plot(self, skip_start:int=10, skip_end:int=1, sma:int=5, smooth:bool=False, suggestion:bool=True):
        """Plots learning rate, raw loss, and smoothed loss in one frame.
        
        Args:
            skip_start (int):  lr's to skip from beginning
            skip_end (int):    lr's to skip from end
            sma (int):         moving average step to fetch 'sma' lagged value from the current loss
            smooth (bool):     whether to plot smoothed original loss
            suggestion (bool): show suggestions for learning rate
        """
        lrs = self._split_list(self.cb.lrs, skip_start, skip_end)
        iterations = self._split_list(list(range(len(self.cb.lrs))), skip_start, skip_end)
        
        losses = self._split_list(self.cb.losses, skip_start, skip_end)
        _losses = losses.copy()
        smooth_losses = self._split_list(self.cb.smooth_loss, skip_start, skip_end)
        
        fig, ax = plt.subplots(1,3, figsize=(20,5))
        ax1, ax2, ax3 = ax.ravel()
        
        ax1.plot(iterations, lrs)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate vs. Iterations')
        
        if smooth:
            xs = np.arange(len(losses))
            spl = UnivariateSpline(xs, losses)
            losses = spl(xs)
        ax2.plot(lrs, losses)
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Learning Rate")
        ax2.set_xscale('log')
        ax2.set_title('Learning Rate vs. Raw Loss')
        
        ax3.plot(lrs, smooth_losses)
        ax3.set_ylabel("Loss (Smooth)")
        ax3.set_xlabel("Learning Rate")
        ax3.set_xscale('log')
        ax3.set_title('Learning Rate vs. Loss (Smooth)')
        
        if suggestion:
            print('Suggestions:')
            _, min_lr = self.get_derivatives(sma)
            ml = np.argmin(_losses)
            print(f'Learning rate: {lrs[ml]} at Minimum Loss: {np.min(_losses)}')
            print(f"Learning rate at min. loss divided by 10: {lrs[ml]/10}")
            print(f'Learning rate at min. loss change: {min_lr}')


class LRFinderCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, lr_mult, steps, linear, stop_divergence, beta=0.98, **kwargs):
        self.start_lr = start_lr
        self.lr_mult = lr_mult
        self.linear = linear
        self.stop_divergence = stop_divergence
        self.steps = steps
        self.bestloss = 1e9
        self.avg_loss = 0
        self.curr_lr = 0
        self.beta = beta
        
    def on_train_begin(self, logs):
        self.lrs, self.losses, self.smooth_loss = [], [], []
        # save model weights before fit
        self.model.save_weights('./lrtmp.h5')
        # save actual lr before fit
        self.original_lr = K.get_value(self.model.optimizer.learning_rate)
        # set lr to initial lr
        K.set_value(self.model.optimizer.learning_rate, self.start_lr)
        # track batch updates
        self.iterations = 1
        self.curr_lr = self.start_lr
    
    def on_batch_end(self, batch, logs):
        opt = self.model.optimizer
        loss = logs['loss']
        # computes exponentially weighted average of losses
        self.avg_loss = self.beta * self.avg_loss + (1-self.beta)*loss
        smooth_loss = self.avg_loss / (1 - self.beta**self.iterations)
        self.smooth_loss.append(smooth_loss)
        # save raw losses
        self.losses.append(loss)
        self.lrs.append(self.curr_lr)
        
        # check whether loss explodes and stop training
        if self.stop_divergence:
            if batch > 5 and (math.isnan(loss) or loss > self.bestloss * 4):
                self.model.stop_training = True
                return
        # save the best loss so far
        if loss < self.bestloss:
            self.bestloss = loss
        
        # update lr and prepare for next batch
        self.iterations += 1
        #mult = self.lr_mult * self.iterations if self.linear else self.lr_mult ** self.iterations
        self.curr_lr *= self.lr_mult
        K.set_value(opt.learning_rate, self.curr_lr) #self.start_lr*mult)
        #self.curr_lr = self.start_lr * mult
        
    def on_train_end(self, logs):
        # restore actual model weights and optimizer state
        self.model.load_weights('./lrtmp.h5')
        K.set_value(self.model.optimizer.learning_rate, self.original_lr)
        os.remove('./lrtmp.h5')
        

