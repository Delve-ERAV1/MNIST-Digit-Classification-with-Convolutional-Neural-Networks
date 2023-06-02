# S5

MNIST Digit Classification with Convolutional Neural Networks


## MNIST Data

```python
# Download train & test data
train_loader, test_loader = get_data()
```

[![image.png](https://i.postimg.cc/bwcdvYFZ/image.png)](https://postimg.cc/Wh8sWVVV)

## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

## Train Model
```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)

# New Line
criterion = nn.CrossEntropyLoss()
num_epochs = 20

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion)
  test(model, device, train_loader, criterion)
  scheduler.step()
 ```

## Model Performance
```python
# Plot Final Viz
plot_model_performance()
```

```
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.0657 Batch_id=117 Accuracy=99.06: 100%|██████████| 118/118 [00:24<00:00,  4.82it/s]
Test set: Average loss: 0.0001, Accuracy: 59466/60000 (99.11%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0655 Batch_id=117 Accuracy=99.09: 100%|██████████| 118/118 [00:24<00:00,  4.83it/s]
Test set: Average loss: 0.0001, Accuracy: 59458/60000 (99.10%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0714 Batch_id=117 Accuracy=99.08: 100%|██████████| 118/118 [00:23<00:00,  4.94it/s]
Test set: Average loss: 0.0001, Accuracy: 59458/60000 (99.10%)

Adjusting learning rate of group 0 to 1.0000e-03.
```


[![image.png](https://i.postimg.cc/9fqDXRcw/image.png)](https://postimg.cc/rdMVQp0q)
