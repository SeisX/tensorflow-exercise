# Description
- [Example 1](./tf-dev-summit-tensorboard-tutorial/): Repository forked from [Dandelion Mané's Github Repository](https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial)
- [Example 2](./how_to_use_tensorboard_live/): Repository forked from [Siraj Raval's Github Repository](https://github.com/llSourcell/how_to_use_tensorboard_live/tree/master)

# Tips
- The above two examples are basically the same.
- 在Windows系统下使用时，执行tensorboard命令需要首先切换到事件文件所在的主路径下。[[1]](https://github.com/tensorflow/tensorboard/issues/52#issuecomment-349521326)
    > 例如事件文件保存于E盘下某一路径，如在C盘路径下执行tensorboard命令，无法打开tensorboard对选定时间进行可视化分析。
