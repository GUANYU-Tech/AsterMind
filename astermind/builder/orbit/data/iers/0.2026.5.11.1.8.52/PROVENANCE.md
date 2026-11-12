# 固定版本的地球定向数据溯源

- 来源发行包：`astropy-iers-data`
- 发行版本：`0.2026.5.11.1.8.52`
- 来源 URL：`https://pypi.org/project/astropy-iers-data/0.2026.5.11.1.8.52/`
- Wheel 文件：`astropy_iers_data-0.2026.5.11.1.8.52-py3-none-any.whl`
- Wheel SHA-256：`40c449c35bd8deabc20053f024f3d4c0cbea2947adf620fcbb6f4242b55e2090`
- 许可证：BSD-3-Clause；保存在 `LICENSE.rst` 中

## 保留的文件

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `finals2000A.all` | 3,741,200 | `eeb76193cd43c065b763d78923f8a8ce2f8a62df5d3e1519a3217fac433bdaa6` |
| `ReadMe.finals2000A` | 3,429 | `7c6182cc0fd0cbece39711f648d15e48b49168925602e360a5709c5ccc8d5a12` |
| `Leap_Second.dat` | 1,359 | `6f7bc6a25841bc394f82bdfd5d7bb22ffcd4548ee28e9822f2927a909e4f912f` |

`finals2000A.all` 覆盖 MJD 41684 至 61533。`Leap_Second.dat`
声明的到期日期为 2026-12-27。两者均覆盖项目的规范场景历元
2026-07-20。`ReadMe.finals2000A` 明确了表格各列的解释方式。
Orbit 在使用默认文件前会校验这三个数据文件的哈希，并且绝不在运行时下载数据。
