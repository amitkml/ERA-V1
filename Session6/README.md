# Network Advanced Concepts

Objective is to implement normalization (Batch Normalization, Layer Normalization, Group Normalization) and regularization (L1 Loss and L2 Loss) techniques on MNIST dataset.

## Normalization

When the network gets trained, intermediate values might be large and can cause imbalanced gradients. Normalization is a technique to convert the data points to a standard scale to **avoid the exploding gradient** problem.

Basic idea is to find mean and standard deviation of the data points for calculating the normalized value of a data point.

    x = x- mean/std

In brief:

|    Normalization    | Representation                                               | Explanation                                                  | Batch Size Dependency | Trainable Parameters           | Hyperparameter dependency | Sample PyTorch Implementation                  | Remarks                           |
| :-----------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------- | ------------------------------ | ------------------------- | ---------------------------------------------- | --------------------------------- |
| Batch Normalization | ![image](https://user-images.githubusercontent.com/17870236/121501907-48c96880-c9fd-11eb-8639-38de0f870686.png) | Rescaling the data points w.r.t each channel                 | No                    | 2 - gamma and beta per channel | No                        | nn.BatchNorm2d(<no of channels>, affine=False) | Most commonly use normalization   |
| Layer Normalization | ![image](https://user-images.githubusercontent.com/17870236/121501955-5383fd80-c9fd-11eb-8608-da3df435112e.png) | Rescaling the data points w.r.t each image across all channels | Yes                   |                                | No                        | nn.GroupNorm(**1**, no of channels)            | Mostly used in lstm networks      |
| Group Normalization | ![image](https://user-images.githubusercontent.com/17870236/121502012-60a0ec80-c9fd-11eb-9e72-203c452f35bb.png) | Rescaling the data points w.r.t specific group of layer in an image | Yes                   |                                | Yes                       | nn.GroupNorm(no of groups, no of channels)     | Works well for smaller batch size |

Let us see each one in detail:

![image](https://user-images.githubusercontent.com/17870236/121506378-4537e080-ca01-11eb-845f-41aa9b76b906.png)


## Batch Normalization

![image](https://user-images.githubusercontent.com/17870236/121502397-c5f4dd80-c9fd-11eb-82c5-c712c20606dd.png)

## Layer Normalization

![image](https://user-images.githubusercontent.com/17870236/121504407-9d6de300-c9ff-11eb-80dc-747b2d48edad.png)


## Group Normalization

![image](https://user-images.githubusercontent.com/17870236/121504026-410ac380-c9ff-11eb-858d-c0481485d182.png)





## Experiments and Model performance

All experimentations are in experiment folder.

- Batch Size - 64 
- Dropout   - 0.03 
- Scheduler - OneCycleLR

|Regularization|	Best Train Accuracy	| Best Test Accuracy |	Best Test Loss| L1 Factor | L2 Factor|
|------------|-----------------|-------------|----------|---|---|
|LayerNorm|98.80|99.48|0.0174|0|0|
|GroupNorm|98.84|99.51|0.0156|0|0|
|BatchNorm|98.59|99.58|0.0151|0|0|
|BatchNorm with L1 |98.02|99.26|0.0217|0.001|0|
|GroupNorm with L1|98.12|99.37|0.0283|0.001|0|
|LayerNorm with L2|98.94|99.56|0.0159|0|0.001|
|BatchNorm with L1 and L2|98.07|99.31|0.0233|0.001|0.001|

### Observations
- When we apply LayerNorm, GroupNorm and BatchNorm techniques individually, without any regularization we observe that BatchNorm performs better than other two techniques.
- Batch norm performed better with higher batch sizes
- We found that GroupNorm was better than batch norm when we have smaller batch size.
- Layernorm performance was lower compared to other two techniques, however we could see some improvement when used with regularization.



## References

- [Group Normalization](https://www.youtube.com/watch?v=l_3zj6HeWUE&t=430s)
- [Running Google Colab with VSCode](https://eide.ai/vscode/2020/09/14/colab-vscode-gpu.html)
- [Pytorch Layernorm Implementation](https://discuss.pytorch.org/t/is-there-a-layer-normalization-for-conv2d/7595/3)
- [Hyperparameter tuning and experimenting](https://deeplizard.com/learn/video/ycxulUVoNbk)

