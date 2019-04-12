## 版本说明

1. v1: base model
2. v2:增加feats之后的dropout，把attention的concat改成加法
3. v3:在sub层增加一个宽度为9的kernel
4. v4:在 conv 5的上面增加一个 kernel size为3的max pooling层
5. v5:去掉sub的conv层
参见 git log